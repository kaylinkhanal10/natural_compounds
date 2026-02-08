import torch
from molecule_encoder import MoleculeEncoder
from graph_dataset import GraphDataset
from torch_geometric.loader import DataLoader
import argparse
import pandas as pd
import numpy as np
import os
import pickle

def generate_embeddings(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Data (Full dataset for inference, no split)
    dataset = GraphDataset(args.data_path, split='all')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load Model
    encoder = MoleculeEncoder(node_dim=10, edge_dim=4, hidden_dim=128, out_dim=128).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        encoder.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print("Warning: No checkpoint found. Using random initialization (for testing).")
    
    encoder.eval()
    
    results = []
    
    print("Generating Embeddings...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            emb = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Map back to original indices to get metadata (CHEMBL ID)
            # The loader preserves order if shuffle=False
            batch_size = emb.size(0)
            start_idx = i * args.batch_size
            
            emb_np = emb.cpu().numpy()
            
            for j in range(batch_size):
                idx = start_idx + j
                if idx < len(dataset):
                    # Get metadata from dataframe
                    row = dataset.df.iloc[dataset.indices[idx]]
                    results.append({
                        'chembl_id': row.get('chembl_id', f'UNK_{idx}'),
                        'smiles': row['smiles'],
                        'embedding': emb_np[j]
                    })
                    
    # Save
    df_emb = pd.DataFrame(results)
    if args.output.endswith('.pkl'):
        with open(args.output, 'wb') as f:
            pickle.dump(results, f)
    else:
        # Save as Parquet or CSV (embeddings as strings) is messy
        # Pickle is best for numpy arrays
        with open(args.output, 'wb') as f:
            pickle.dump(results, f)
            
    print(f"Saved embeddings for {len(results)} molecules to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='backend/app/ml/world_model/checkpoints/encoder_pretrained.pt')
    parser.add_argument('--output', type=str, default='raw_data/chembl_embeddings.pkl')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    generate_embeddings(args)
