import torch
import sys
import os

# Fix path
sys.path.append(os.getcwd())

from backend.app.ml.world_model.molecule_encoder import MoleculeEncoder
from backend.app.ml.world_model.biology_heads import TargetPredictionHead, ActivityRegressionHead

def test_models():
    print("Testing Models...")
    
    # 1. Encoder
    node_dim = 11
    edge_dim = 4
    hidden = 128
    
    encoder = MoleculeEncoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden)
    
    # Dummy Batch
    # 2 molecules
    # Mol 1: 3 atoms, 2 bonds
    # Mol 2: 2 atoms, 1 bond
    x = torch.randn(5, node_dim)
    edge_index = torch.tensor([[0, 1, 1, 0, 3, 4, 4, 3],
                               [1, 0, 0, 1, 4, 3, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(8, edge_dim)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    
    emb = encoder(x, edge_index, edge_attr, batch)
    print(f"Encoder Output Shape: {emb.shape}")
    assert emb.shape == (2, 128)
    
    # 2. Heads
    num_targets = 50
    target_head = TargetPredictionHead(hidden, num_targets)
    logits = target_head(emb)
    print(f"Target Head Output Shape: {logits.shape}")
    assert logits.shape == (2, num_targets)
    
    act_head = ActivityRegressionHead(hidden)
    act = act_head(emb)
    print(f"Activity Head Output Shape: {act.shape}")
    assert act.shape == (2, 1)
    
    print("All Model Tests Passed.")

if __name__ == "__main__":
    test_models()
