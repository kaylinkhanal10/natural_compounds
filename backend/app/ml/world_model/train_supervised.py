import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from molecule_encoder import MoleculeEncoder
from biology_heads import TargetPredictionHead
from chembl_dataset import ChEMBLDataset
from metrics import compute_retrieval_metrics
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def train(epoch, encoder, head, loader, optimizer, criterion, device):
    encoder.train()
    head.train()
    total_loss = 0
    total_recall = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward
        # Encoder: Graph -> Embedding [batch, 128]
        emb = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # Head: Embedding -> Target Logits [batch, num_targets]
        logits = head(emb)
        
        # Targets
        # batch.y_target is [batch, 1, num_targets], squeeze it
        targets = batch.y_target.squeeze(1).to(device)
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Metrics (on train mostly for sanity, val is real test)
        with torch.no_grad():
             m = compute_retrieval_metrics(targets, logits, k=10)
             total_recall += m['recall_at_k']
            
    avg_loss = total_loss / len(loader)
    avg_recall = total_recall / len(loader)
    return avg_loss, avg_recall

def validate(encoder, head, loader, criterion, device):
    encoder.eval()
    head.eval()
    total_loss = 0
    total_recall = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            emb = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits = head(emb)
            targets = batch.y_target.squeeze(1).to(device)
            
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
            m = compute_retrieval_metrics(targets, logits, k=10)
            total_recall += m['recall_at_k']
            
    return total_loss / len(loader), total_recall / len(loader)

def plot_metrics(history, output_dir):
    df = pd.DataFrame(history)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
    plt.close()
    
    # Recall
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_recall'], label='Train Recall@10')
    plt.plot(df['epoch'], df['val_recall'], label='Val Recall@10')
    plt.title('Retrieval Performance (Recall@10)')
    plt.xlabel('Epoch')
    plt.ylabel('Recall@10')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'retrieval_at_k.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='/mnt/natural_medicine_data/chembl_36/chembl_36_sqlite/chembl_36.db')
    parser.add_argument('--limit', type=int, default=None, help="Limit number of compounds for debugging")
    parser.add_argument('--output_dir', type=str, default='experiments/exp_001_encoder_training', help="Directory to save artifacts")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save Config
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Dataset
    print("Initializing ChEMBL Dataset...")
    dataset = ChEMBLDataset(args.data_path, split='train', limit=args.limit)
    val_dataset = ChEMBLDataset(args.data_path, split='val', limit=args.limit)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_targets = dataset.num_targets
    print(f"Number of Targets: {num_targets}")
    
    if num_targets == 0:
        print("Error: No targets found. Exiting.")
        return

    # 2. Model
    # Node features: 10 (9 types + other) + 1 (Aromatic) = 11
    # Edge features: 4
    encoder = MoleculeEncoder(node_dim=11, edge_dim=4, hidden_dim=128, out_dim=128).to(device)
    
    head = TargetPredictionHead(input_dim=128, num_targets=num_targets).to(device)
    
    # 3. Optimization
    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Supervised Training...")
    
    best_recall = 0.0
    metrics_history = []
    
    # Checkpoints stored in experiment dir
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        train_loss, train_recall = train(epoch, encoder, head, train_loader, optimizer, criterion, device)
        val_loss, val_recall = validate(encoder, head, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Recall@10: {train_recall:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Recall@10: {val_recall:.4f}")
              
        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_recall': train_recall,
            'val_loss': val_loss,
            'val_recall': val_recall
        })
        
        # Save metrics JSON incrementally
        with open(os.path.join(args.output_dir, 'eval_metrics.json'), 'w') as f:
            json.dump(metrics_history, f, indent=4)
            
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save({
                'encoder': encoder.state_dict(),
                'head': head.state_dict(),
                'target_map': dataset.target_map
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            print("--> Best Model Saved.")

    # Save Final Plots
    plot_metrics(metrics_history, args.output_dir)
    print("Training Complete. Artifacts saved.")

if __name__ == "__main__":
    main()
