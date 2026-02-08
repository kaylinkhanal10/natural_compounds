import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from molecule_encoder import MoleculeEncoder
from graph_dataset import GraphDataset
import os
import argparse
import pandas as pd

def train(epoch, model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward
        # x, edge_index, edge_attr, batch_index
        embedding = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # Predict Properties (Simple Linear Probe on top of embedding for pre-training)
        # We need a "Head" here. 
        # For simplicity in this script, we'll assume the model output IS the prediction 
        # OR we add a temporary head.
        # Let's add a projection head here or use the model's output if dim matches.
        # Embedding is 128. Properties are 8.
        # We needs a head.
        pass # Handle in main loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='raw_data/chembl_36_extracted.xlsx')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Check data
    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}")
        return

    # Dataset
    dataset = GraphDataset(args.data_path, split='train')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Calculate Mean/Std for targets (from dataframe directly)
    # y columns: mw, logp, tpsa, hba, hbd, rotb, nring, nrom
    prop_cols = ['mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotb', 'nring', 'nrom']
    
    # Fix: Ensure columns align with dataset item order, but mean/std can be global
    df_train = dataset.df.iloc[dataset.indices]
    
    # Simple imputation for NaN
    targets = df_train[prop_cols].fillna(df_train[prop_cols].mean())
    
    y_mean = torch.tensor(targets.mean().values, dtype=torch.float).to(device)
    y_std = torch.tensor(targets.std().values, dtype=torch.float).to(device)
    
    # Avoid div by zero
    y_std[y_std == 0] = 1.0
    
    print(f"Target Normalization -- Mean: {y_mean[:3]}...")
    
    # Model
    # Node features: 10 (9 types + other)
    # Edge features: 4
    encoder = MoleculeEncoder(node_dim=10, edge_dim=4, hidden_dim=128, out_dim=128).to(device)
    
    # Property Head (8 props)
    prop_head = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 8)
    ).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(prop_head.parameters()), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("Starting Pre-training (Property Reconstruction)...")
    
    for epoch in range(args.epochs):
        encoder.train()
        prop_head.train()
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            emb = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = prop_head(emb)
            
            # Normalize Targets on the fly
            targets_norm = (batch.y.to(device) - y_mean) / y_std
            
            loss = criterion(pred, targets_norm)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")
        
    # Save
    os.makedirs('backend/app/ml/world_model/checkpoints', exist_ok=True)
    torch.save(encoder.state_dict(), 'backend/app/ml/world_model/checkpoints/encoder_pretrained.pt')
    print("Saved Pre-trained Encoder.")

if __name__ == "__main__":
    main()
