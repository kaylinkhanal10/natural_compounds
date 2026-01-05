import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import yaml

class MMVAEDataset(Dataset):
    def __init__(self, data_dir='raw_data', config_path='backend/app/ml/world_model/config.yaml', split='train'):
        self.data_dir = data_dir
        self.split = split
        
        # Load chemical data
        chem_path = os.path.join(data_dir, 'chemical_property.xlsx')
        self.chem_df = pd.read_excel(chem_path)
        self.chem_df.columns = self.chem_df.columns.str.lower()
        
        # Load protein data
        prot_path = os.path.join(data_dir, 'chemical_protein.xlsx')
        self.prot_df = pd.read_excel(prot_path)
        self.prot_df.columns = self.prot_df.columns.str.lower()
        
        # Resolve compounds (strict deduplication)
        self._resolve_identities()
        
        # Build features
        self._build_chemical_features()
        self._build_protein_features()
        
        # Split data
        self._split_data()
        
        # Save feature metadata for inference
        self.feature_metadata = {
            'chem_cols': self.chem_feature_cols,
            'prot_vocab': self.protein_vocab,
            'chem_mean': self.chem_mean,
            'chem_std': self.chem_std
        }

    def _resolve_identities(self):
        # We need a stable list of unique compounds.
        # Primary Key: inchikey, Fallback: compound_id
        # We will use the index of this unique list as the dataset index.
        
        # Simplified: Use inchikey if present, else dropped for this VAE (strict)
        self.chem_df = self.chem_df.dropna(subset=['inchikey']).drop_duplicates(subset=['inchikey']).reset_index(drop=True)
        self.compound_ids = self.chem_df['inchikey'].tolist()
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.compound_ids)}
        
    def _build_chemical_features(self):
        # Numeric columns to use
        # Dynamic mapping based on presence
        candidates = [
            'mw', 'exact_mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotb', 
            'atom_count', 'nring', 'nrom'
        ]  
        # Match case-insensitive
        cols = self.chem_df.columns
        lower_cols = {c.lower(): c for c in cols}
        
        self.chem_feature_cols = []
        for c in candidates:
            if c in lower_cols:
                self.chem_feature_cols.append(lower_cols[c])
        
        X = self.chem_df[self.chem_feature_cols].copy()
        
        # Cleaning
        # 1. Coerce to numeric
        for c in self.chem_feature_cols:
            X[c] = pd.to_numeric(X[c], errors='coerce')
            
        # 2. Impute (Median)
        self.chem_mean = X.median()
        X = X.fillna(self.chem_mean)
        
        # 3. Clip Outliers (1-99%)
        lower = X.quantile(0.01)
        upper = X.quantile(0.99)
        X = X.clip(lower, upper, axis=1)
        
        # 4. Z-score Normalize
        self.chem_std = X.std()
        # Handle zero std
        self.chem_std[self.chem_std == 0] = 1.0
        X = (X - self.chem_mean) / self.chem_std
        
        self.X_chem = X.values.astype(np.float32)
        
    def _build_protein_features(self):
        # Build multi-hot vectors
        # Map inchikey -> list of protein IDs
        
        # Assuming prot_df has keys linking to compounds. Column map needed?
        # Let's inspect column map or assume standard keys if not provided.
        # Based on previous context, 'inchikey' or 'compound_id' links them.
        
        # Let's assume 'inchikey' exists in prot_df or we link via 'compound_id'
        # Fallback: strict mapping via 'compound_id' if inchikey missing in protein file
        
        # Unique proteins
        if 'protein_id' in self.prot_df.columns:
            prot_col = 'protein_id'
        elif 'proteinid' in self.prot_df.columns:
            prot_col = 'proteinid'
        else:
            prot_col = 'target_id' # adjust as needed
            
        # Get unique proteins sorted
        self.protein_vocab = sorted(self.prot_df[prot_col].unique().astype(str))
        self.prot_to_idx = {p: i for i, p in enumerate(self.protein_vocab)}
        
        num_compounds = len(self.compound_ids)
        num_proteins = len(self.protein_vocab)
        self.X_prot = np.zeros((num_compounds, num_proteins), dtype=np.float32)
        
        # Group proteins by compound
        # We need a link. If protein file has 'compound_id' and chem file has 'source_id'/'compound_id'
        # We use the map built in ingestion. 
        # For this standalone module, we might rely on 'compound_id' present in both.
        
        # Optimization: Create map structure 
        # compound_id (or inchikey) -> [prot_idx]
        
        # Check alignment:
        # Chem df: inchikey, compound_id
        # Prot df: compound_id (likely)
        
        # Allow linking logic:
        # Map Prot `compound_id` -> Chem `inchikey` using Chem DF lookup
        # Check available columns for ID
        if 'compound_id' in self.chem_df.columns:
            id_col = 'compound_id'
        elif 'id' in self.chem_df.columns:
            id_col = 'id'
        else:
            id_col = 'source_id' # fallback
            
        cid_to_inchi = pd.Series(self.chem_df['inchikey'].values, index=self.chem_df[id_col].values).to_dict()
        
        valid_links = 0
        for _, row in self.prot_df.iterrows():
            cid = row.get('compound_id')
            pid = str(row.get(prot_col))
            
            if cid in cid_to_inchi:
                 inchi = cid_to_inchi[cid]
                 if inchi in self.id_to_idx and pid in self.prot_to_idx:
                     c_idx = self.id_to_idx[inchi]
                     p_idx = self.prot_to_idx[pid]
                     self.X_prot[c_idx, p_idx] = 1.0
                     valid_links += 1
                         
    def _split_data(self):
        # 80/10/10
        N = len(self.X_chem)
        indices = np.random.permutation(N)
        train_end = int(0.8 * N)
        val_end = int(0.9 * N)
        
        self.train_idx = indices[:train_end]
        self.val_idx = indices[train_end:val_end]
        self.test_idx = indices[val_end:]
        
    def __len__(self):
        if self.split == 'train':
            return len(self.train_idx)
        elif self.split == 'val':
            return len(self.val_idx)
        else:
            return len(self.test_idx)
            
    def __getitem__(self, idx):
        if self.split == 'train':
            real_idx = self.train_idx[idx]
        elif self.split == 'val':
            real_idx = self.val_idx[idx]
        else:
            real_idx = self.test_idx[idx]
            
        x_chem = torch.tensor(self.X_chem[real_idx])
        x_prot = torch.tensor(self.X_prot[real_idx])
        return x_chem, x_prot

