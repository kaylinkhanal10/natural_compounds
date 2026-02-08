import torch
import yaml
import os
import sys
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch

# Ensure path
sys.path.append(os.getcwd())

from app.ml.world_model.molecule_encoder import MoleculeEncoder

class SupervisedInference:
    def __init__(self, config_path='backend/app/ml/world_model/config.yaml', checkpoint_path=None):
        if not os.path.exists(config_path):
             # Try relative
             if os.path.exists(f"../{config_path}"):
                 config_path = f"../{config_path}"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Model
        # Node dim 11, Edge dim 4 (Hardcoded matching training)
        self.encoder = MoleculeEncoder(node_dim=11, edge_dim=4, hidden_dim=self.config['model']['hidden_chem']).to(self.device)
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config['train']['save_dir'], 'best_supervised.pt')
            
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Checkpoint contains 'encoder' key
            if 'encoder' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder'])
            else:
                self.encoder.load_state_dict(checkpoint) # Fallback
            self.encoder.eval()
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

    def _smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        x = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            feats = [0] * (len(atom_types) + 1)
            if sym in atom_types:
                feats[atom_types.index(sym)] = 1
            else:
                feats[-1] = 1 
            feats.append(1 if atom.GetIsAromatic() else 0)
            x.append(feats)
        x = torch.tensor(x, dtype=torch.float)
        
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
            edge_indices.append([j, i])
            edge_attrs.append(bfeat)
            
        if not edge_indices:
             return None 
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def encode(self, smiles_list):
        graphs = []
        valid_indices = []
        
        for i, smi in enumerate(smiles_list):
            if not smi:
                continue
            g = self._smiles_to_graph(smi)
            if g:
                graphs.append(g)
                valid_indices.append(i)
                
        if not graphs:
            return np.array([])
            
        # Batching
        batch_size = 32
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(graphs), batch_size):
                batch_graphs = Batch.from_data_list(graphs[i:i+batch_size]).to(self.device)
                emb = self.encoder(batch_graphs.x, batch_graphs.edge_index, batch_graphs.edge_attr, batch_graphs.batch)
                embeddings.append(emb.cpu().numpy())
                
        full_embs = np.concatenate(embeddings, axis=0)
        
        # Remap to original list (fill failures with zeros or exclude)
        # We return a list of same length, None if failed
        result = [None] * len(smiles_list)
        for idx, valid_idx in enumerate(valid_indices):
            result[valid_idx] = full_embs[idx]
            
        return result
