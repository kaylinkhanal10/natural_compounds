import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import os
import json
from .model import MultiModalVAE
from .dataset import MMVAEDataset
from sklearn.metrics.pairwise import cosine_similarity

class WorldModelInference:
    def __init__(self, config_path='backend/app/ml/world_model/config.yaml', checkpoint_path=None):
        self.config_path = config_path.replace('backend/app/ml/world_model/', '') # Adjust relative if needed
        if os.path.isabs(config_path):
             self.config_path = config_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Dataset Metadata (using Dataset class to ensure consistent feature building)
        # In a production setting, we would load just the JSON metadata.
        # Here we re-instantiate dataset to get features (fast enough for this scale).
        self.dataset = MMVAEDataset(data_dir=self.config['data']['raw_dir'], config_path=config_path, split='train')
        
        self.chem_dim = self.dataset.X_chem.shape[1]
        self.prot_dim = self.dataset.X_prot.shape[1]
        
        # Initialize Model
        self.model = MultiModalVAE(chem_dim=self.chem_dim, prot_dim=self.prot_dim, latent_dim=self.config['model']['latent_dim']).to(self.device)
        
        # Load Checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config['train']['save_dir'], 'best_model.pt')
            
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Loaded World Model from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

        # Pre-compute all embeddings for efficient search
        self.embeddings = None
        self.compound_ids = self.dataset.compound_ids
        self._compute_all_embeddings()

    def _compute_all_embeddings(self):
        embeddings = []
        batch_size = 256
        
        # We need to compute for ALL compounds, not just train split
        # Re-using the full X_chem and X_prot from dataset object
        X_chem = torch.tensor(self.dataset.X_chem).to(self.device)
        X_prot = torch.tensor(self.dataset.X_prot).to(self.device)
        
        with torch.no_grad():
            for i in range(0, len(X_chem), batch_size):
                b_chem = X_chem[i:i+batch_size]
                b_prot = X_prot[i:i+batch_size]
                
                # Encode
                h_chem = self.model.chem_encoder(b_chem)
                h_prot = self.model.prot_encoder(b_prot)
                h_fused = torch.cat([h_chem, h_prot], dim=1)
                h_latent = self.model.fusion_net(h_fused)
                mu = self.model.fc_mu(h_latent)
                # We use mu (mean) as the deterministic embedding
                
                embeddings.append(mu.cpu().numpy())
                
        self.embeddings = np.concatenate(embeddings, axis=0)

    def get_embedding(self, inchikey):
        if inchikey in self.dataset.id_to_idx:
            idx = self.dataset.id_to_idx[inchikey]
            return self.embeddings[idx]
        return None

    def find_nearest_neighbors(self, inchikey, k=10):
        target_emb = self.get_embedding(inchikey)
        if target_emb is None:
            return []
            
        target_emb = target_emb.reshape(1, -1)
        sims = cosine_similarity(target_emb, self.embeddings)[0]
        
        # Top-k
        indices = np.argsort(sims)[::-1][1:k+1] # Skip self
        
        results = []
        for idx in indices:
            res = {
                'inchikey': self.compound_ids[idx],
                'similarity': float(sims[idx])
            }
            results.append(res)
            
        return results

    def save_embeddings_to_csv(self, output_path='embeddings.csv'):
        df = pd.DataFrame(self.embeddings)
        df.index = self.compound_ids
        df.to_csv(output_path)
