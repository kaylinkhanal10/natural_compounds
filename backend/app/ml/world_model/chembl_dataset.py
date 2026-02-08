import torch
from torch.utils.data import Dataset
import pandas as pd
import sqlite3
import numpy as np
import os
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import SaltRemover
from torch_geometric.data import Data

class ChEMBLDataset(Dataset):
    def __init__(self, sqlite_path, split='train', limit=None, cache_dir='backend/data/cache'):
        """
        Args:
            sqlite_path: Path to chembl_36.db
            split: 'train', 'val', or 'test'
            limit: Optional limit for testing initial query
            cache_dir: Directory to save processed pickle files
        """
        self.sqlite_path = sqlite_path
        self.split = split
        self.limit = limit
        self.cache_dir = cache_dir
        
        # Target mapping (chembl_id -> integer index)
        self.target_map = {} 
        self.num_targets = 0
        
        self.processed_data = []
        
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f'chembl_processed_{split}_{limit if limit else "full"}.pkl')
        map_file = os.path.join(self.cache_dir, 'target_map.pkl')
        
        if os.path.exists(cache_file) and os.path.exists(map_file):
            print(f"Loading cached dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.processed_data = pickle.load(f)
            with open(map_file, 'rb') as f:
                self.target_map = pickle.load(f)
                self.num_targets = len(self.target_map)
        else:
            if not os.path.exists(sqlite_path):
                print(f"Warning: ChEMBL DB not found at {sqlite_path}. Dataset will be empty.")
            else:
                self._process_and_cache(cache_file, map_file)

    def _process_and_cache(self, cache_file, map_file):
        print("Querying ChEMBL SQLite...")
        conn = sqlite3.connect(self.sqlite_path)
        
        # Filter: Standard units, high confidence, Single Protein
        # We group by InChIKey to merge salts/stereoisomers where appropriate for bioactivity
        query = """
        SELECT 
            cs.canonical_smiles,
            cs.standard_inchi_key,
            td.chembl_id as target_chembl_id,
            act.pchembl_value,
            ass.confidence_score
        FROM activities act
        JOIN compound_structures cs ON act.molregno = cs.molregno
        JOIN assays ass ON act.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        WHERE act.pchembl_value IS NOT NULL
        AND act.standard_type IN ('IC50', 'Ki', 'EC50', 'Kd')
        AND td.target_type = 'SINGLE PROTEIN'
        AND ass.confidence_score >= 7
        """
        
        if self.limit:
            query += f" LIMIT {self.limit * 10}" # Fetch more rows to group
            
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("No data found matching criteria.")
            return

        # 1. Build Target Vocabulary (Top N targets?)
        # For V1, let's take all targets that appear more than K times to avoid sparsity
        target_counts = df['target_chembl_id'].value_counts()
        valid_targets = target_counts[target_counts >= 10].index.tolist()
        
        self.target_map = {tid: i for i, tid in enumerate(valid_targets)}
        self.num_targets = len(valid_targets)
        print(f"Found {self.num_targets} valid targets (freq >= 10)")
        
        # Save map
        with open(map_file, 'wb') as f:
            pickle.dump(self.target_map, f)
            
        # 2. Group by Compound (InChIKey)
        # Check standard_inchi_key presence
        # If missing, fallback to SMILES
        print("Grouping by compound...")
        grouped = df.groupby('standard_inchi_key')
        
        remover = SaltRemover.SaltRemover()
        
        valid_compounds = []
        
        for inchi_key, group in tqdm(grouped, desc="Processing Compounds"):
            # Get canonical smiles (first one)
            raw_smiles = group.iloc[0]['canonical_smiles']
            
            # Preprocess Structure
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None: 
                continue
                
            # Remove Salts
            mol = remover.StripMol(mol)
            # Canonicalize
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) # Flatten stereo for base activity
            
            # Collect Targets
            # Get list of targets in our map
            compound_targets = set()
            for tid in group['target_chembl_id']:
                if tid in self.target_map:
                    compound_targets.add(self.target_map[tid])
                    
            if not compound_targets:
                continue
                
            # Max pChEMBL (optional)
            max_activity = group['pchembl_value'].max()
            
            valid_compounds.append({
                'smiles': smiles,
                'inchi_key': inchi_key,
                'targets': list(compound_targets),
                'pchembl_max': max_activity
            })
            
            if self.limit and len(valid_compounds) >= self.limit:
                break
                
        # Split
        np.random.seed(42)
        np.random.shuffle(valid_compounds)
        
        N = len(valid_compounds)
        train_end = int(0.8 * N)
        val_end = int(0.9 * N)
        
        if self.split == 'train':
            data_split = valid_compounds[:train_end]
        elif self.split == 'val':
            data_split = valid_compounds[train_end:val_end]
        else:
            data_split = valid_compounds[val_end:]
            
        self.processed_data = data_split
        
        with open(cache_file, 'wb') as f:
            pickle.dump(self.processed_data, f)
            
        print(f"Processed and cached {len(self.processed_data)} compounds for split {self.split}")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        smiles = item['smiles']
        
        # Create Graph
        graph = self._smiles_to_graph(smiles)
        
        if graph is None:
            # Should not happen if correctly processed, but safety
            return self.__getitem__((idx + 1) % len(self))
            
        # Targets Multi-Hot
        y_target = torch.zeros(self.num_targets, dtype=torch.float)
        y_target[item['targets']] = 1.0
        
        # Attach to graph
        graph.y_target = y_target.unsqueeze(0) # [1, num_targets]
        graph.y_pchembl = torch.tensor([item['pchembl_max']], dtype=torch.float).unsqueeze(0)
        
        return graph

    def _smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Node features (Simple: Symbol + IsAromatic)
        # Can expand to hybridization etc.
        atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        
        x = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            feats = [0] * (len(atom_types) + 1)
            if sym in atom_types:
                feats[atom_types.index(sym)] = 1
            else:
                feats[-1] = 1 # Other
            
            # Aromatic
            feats.append(1 if atom.GetIsAromatic() else 0)
            
            x.append(feats)
            
        x = torch.tensor(x, dtype=torch.float) # [N, node_dim]
        
        # Edge features
        edge_indices = []
        edge_attrs = []
        
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
                      
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            btype = bond.GetBondType()
            bfeat = [0] * len(bond_types)
            if btype in bond_types:
                bfeat[bond_types.index(btype)] = 1
                
            edge_indices.append([i, j])
            edge_attrs.append(bfeat)
            
            # Undirected
            edge_indices.append([j, i])
            edge_attrs.append(bfeat)
            
        if not edge_indices:
             return None # Single atom?
             
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
