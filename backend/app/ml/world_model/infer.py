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
    def __init__(self, config_path='app/ml/world_model/config.yaml', checkpoint_path=None):
        self.config_path = config_path.replace('backend/app/ml/world_model/', '') # Adjust relative if needed
        if os.path.isabs(config_path):
             self.config_path = config_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu') # Forced due to RTX 5090 incompatibility
        
        # Load Dataset Metadata (using Dataset class to ensure consistent feature building)
        # In a production setting, we would load just the JSON metadata.
        # Here we re-instantiate dataset to get features (fast enough for this scale).
        self.dataset = MMVAEDataset(data_dir=self.config['data']['raw_dir'], config_path=config_path, split='train')
        
        self.chem_dim = self.dataset.X_chem.shape[1]
        self.chem_dim = self.dataset.X_chem.shape[1]
        
        # Initialize Model
        # Initialize Model
        self.model = MultiModalVAE(chem_dim=self.chem_dim, latent_dim=self.config['model']['latent_dim']).to(self.device)
        
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
        # Re-using the full X_chem from dataset object
        X_chem = torch.tensor(self.dataset.X_chem).to(self.device)
        
        with torch.no_grad():
            for i in range(0, len(X_chem), batch_size):
                b_chem = X_chem[i:i+batch_size]
                
                # Encode (Chemistry Only)
                h_chem = self.model.chem_encoder(b_chem)
                mu = self.model.fc_mu(h_chem)
                # We use mu (mean) as the deterministic embedding
                
                embeddings.append(mu.cpu().numpy())
                
        self.embeddings = np.concatenate(embeddings, axis=0)

    def get_embedding(self, inchikey):
        if inchikey in self.dataset.id_to_idx:
            idx = self.dataset.id_to_idx[inchikey]
            return self.embeddings[idx]
        return None



    def find_compounds_by_latent(self, latent_vector, k=10):
        """
        Finds k nearest compounds to a given latent vector.
        """
        if latent_vector is None:
            return []
            
        target = latent_vector.reshape(1, -1)
        sims = cosine_similarity(target, self.embeddings)[0]
        
        indices = np.argsort(sims)[::-1][:k] # Include self/top 1
        
        results = []
        for idx in indices:
            res = {
                'inchikey': self.compound_ids[idx],
                'similarity': float(sims[idx])
            }
            results.append(res)
            
        return results

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

    def compare_herb_profiles(self, compounds_a_ids, compounds_b_ids):
        """
        Computes the latent distance between two herbs based on their compound constituents.
        Returns:
            - centroid_distance (0.0 to 1.0): Cosine distance between mean vectors.
            - complementarity_score (0.0 to 1.0): Interpreted score.
        """
        # 1. Gather Embeddings
        emb_a = []
        for cid in compounds_a_ids:
            vec = self.get_embedding(cid)
            if vec is not None:
                emb_a.append(vec)
                
        emb_b = []
        for cid in compounds_b_ids:
            vec = self.get_embedding(cid)
            if vec is not None:
                emb_b.append(vec)
                
        # Handle missing data
        if not emb_a or not emb_b:
            return None
            
        # 2. Compute Centroids (Mean of all compounds in the herb)
        # Shape: (latent_dim,)
        centroid_a = np.mean(np.vstack(emb_a), axis=0)
        centroid_b = np.mean(np.vstack(emb_b), axis=0)
        
        # 3. Compute Cosine Similarity & Distance
        # Reshape for sklearn: (1, dim)
        sim = cosine_similarity(centroid_a.reshape(1, -1), centroid_b.reshape(1, -1))[0][0]
        
        # Distance = 1 - Similarity (Range: 0 to 2 for cosine, but usually 0-1 for normalized vectors in positive quadrant)
        # VAE space is usually centered around 0, so similarity can be negative (-1 to 1).
        # Distance = (1 - sim) / 2 to normalize to 0-1 range? 
        # Or just use 1 - sim (0 to 2). larger = more distant.
        dist = 1.0 - sim # Range [0, 2] usually.
        
        # 4. Interpret
        # If dist is high (> 0.5), they are complementary.
        # If dist is low (< 0.1), they are redundant.
        
        return {
            "centroid_similarity": float(sim),
            "centroid_distance": float(dist),
            "num_compounds_a": len(emb_a),
            "num_compounds_b": len(emb_b)
        }

    def save_embeddings_to_csv(self, output_path='embeddings.csv'):
        df = pd.DataFrame(self.embeddings)
        df.index = self.compound_ids
        df.to_csv(output_path)
