import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os
from rdkit import Chem
from rdkit.Chem import rdmolops

class GraphDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.split = split
        
        # Load extracted data
        print(f"Loading data from {data_path}...")
        self.df = pd.read_excel(data_path)
        
        # Valid SMILES only
        self.df = self.df.dropna(subset=['smiles']).reset_index(drop=True)
        
        # Simple split (80/10/10) - Deterministic
        N = len(self.df)
        indices = np.arange(N)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_end = int(0.8 * N)
        val_end = int(0.9 * N)
        
        if self.split == 'train':
            self.indices = indices[:train_end]
        elif self.split == 'val':
            self.indices = indices[train_end:val_end]
        else:
            self.indices = indices[val_end:]
            
        print(f"GraphDataset ({split}): {len(self.indices)} samples.")

    def _smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Node Features (Atom Type)
        # Simplified: C, N, O, F, P, S, Cl, Br, I, Other
        atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        
        x = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            feats = [0] * (len(atom_types) + 1)
            if sym in atom_types:
                feats[atom_types.index(sym)] = 1
            else:
                feats[-1] = 1 # Other
            x.append(feats)
            
        x = torch.tensor(x, dtype=torch.float)
        
        # Edge Features (Bond Type)
        # Single, Double, Triple, Aromatic
        edge_indices = []
        edge_attrs = []
        
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            btype = bond.GetBondType()
            bfeat = [0] * 4
            if btype in bond_types:
                bfeat[bond_types.index(btype)] = 1
            
            # Add bidirectional
            edge_indices.append([i, j])
            edge_attrs.append(bfeat)
            edge_indices.append([j, i])
            edge_attrs.append(bfeat)
            
        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]
        smiles = row['smiles']
        
        data = self._smiles_to_graph(smiles)
        if data is None:
            data = self._smiles_to_graph("CC")
            
        # Add Properties as targets (y)
        # Columns: mw, exact_mw, logp, tpsa, hba, hbd, rotb, atom_count, nring, nrom
        # We need to map these to a tensor.
        # Assuming these columns exist in the df (lowercase)
        prop_cols = ['mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotb', 'nring', 'nrom']
        y = []
        for col in prop_cols:
            val = row.get(col, 0.0)
            try:
                val = float(val)
            except:
                val = 0.0
            y.append(val)
            
        data.y = torch.tensor([y], dtype=torch.float)
        
        return data
