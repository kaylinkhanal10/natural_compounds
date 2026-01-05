import torch
import yaml
import os
import json
import numpy as np
import argparse
from torch.utils.data import DataLoader
from .dataset import MMVAEDataset
from .model import MultiModalVAE
from .losses import VAELoss

def train_one_epoch(model, dataloader, criterion, optimizer, beta, device):
    model.train()
    total_loss = 0
    total_chem = 0
    total_prot = 0
    total_kld = 0
    
    for x_chem, x_prot in dataloader:
        x_chem = x_chem.to(device)
        x_prot = x_prot.to(device)
        
        optimizer.zero_grad()
        recon_chem, recon_prot, mu, logvar, z = model(x_chem, x_prot)
        
        loss, l_chem, l_prot, kld = criterion(recon_chem, x_chem, recon_prot, x_prot, mu, logvar, beta)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_chem += l_chem.item()
        total_prot += l_prot.item()
        total_kld += kld.item()
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss, total_chem/len(dataloader), total_prot/len(dataloader), total_kld/len(dataloader)

def validate(model, dataloader, criterion, beta, device):
    model.eval()
    total_loss = 0
    total_chem = 0
    total_prot = 0
    total_kld = 0
    
    with torch.no_grad():
        for x_chem, x_prot in dataloader:
            x_chem = x_chem.to(device)
            x_prot = x_prot.to(device)
            
            recon_chem, recon_prot, mu, logvar, z = model(x_chem, x_prot)
            loss, l_chem, l_prot, kld = criterion(recon_chem, x_chem, recon_prot, x_prot, mu, logvar, beta)
            
            total_loss += loss.item()
            total_chem += l_chem.item()
            total_prot += l_prot.item()
            total_kld += kld.item()
            
    avg_loss = total_loss / len(dataloader)
    return avg_loss, total_chem/len(dataloader), total_prot/len(dataloader), total_kld/len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='backend/app/ml/world_model/config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['train']['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset
    dataset = MMVAEDataset(data_dir=config['data']['raw_dir'], config_path=args.config, split='train')
    val_dataset = MMVAEDataset(data_dir=config['data']['raw_dir'], config_path=args.config, split='val')
    
    train_loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    # Feature sizes
    chem_dim = dataset.X_chem.shape[1]
    prot_dim = dataset.X_prot.shape[1]
    print(f"Chem Dim: {chem_dim}, Prot Dim: {prot_dim}")
    
    # Calculate positive weights for protein loss (sparsity handling)
    # Global calc for simplicity
    prot_all = torch.tensor(dataset.X_prot)
    num_ones = prot_all.sum()
    num_zeros = prot_all.numel() - num_ones
    if num_ones > 0:
        pos_weight = (num_zeros / num_ones)
    else:
        pos_weight = 1.0
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    print(f"Protein Pos Weight: {pos_weight:.2f}")

    # 2. Model
    model = MultiModalVAE(chem_dim=chem_dim, prot_dim=prot_dim, latent_dim=config['model']['latent_dim']).to(device)
    
    # 3. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    criterion = VAELoss(lambda_chem=config['train']['lambda_chem'], lambda_prot=config['train']['lambda_prot'], pos_weight=pos_weight_tensor).to(device)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    epochs = config['train']['epochs']
    warmup = config['train']['beta_warmup_epochs']
    
    metrics_history = []
    
    for epoch in range(epochs):
        # Linear Beta Warmup
        if epoch < warmup:
            beta = epoch / warmup
        else:
            beta = 1.0
            
        train_loss, t_chem, t_prot, t_kld = train_one_epoch(model, train_loader, criterion, optimizer, beta, device)
        val_loss, v_chem, v_prot, v_kld = validate(model, val_loader, criterion, beta, device)
        
        print(f"Epoch {epoch+1}/{epochs} | Beta: {beta:.2f} | Train Loss: {train_loss:.4f} (C:{t_chem:.2f}, P:{t_prot:.2f}, K:{t_kld:.2f}) | Val Loss: {val_loss:.4f} (C:{v_chem:.2f}, P:{v_prot:.2f}, K:{v_kld:.2f})")
        
        metrics_history.append({
            'epoch': epoch+1,
            'train_loss': train_loss, 'val_loss': val_loss,
            'val_chem': v_chem, 'val_prot': v_prot
        })
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], 'best_model.pt'))
            print("  --> Saved Best Model")
        else:
            patience_counter += 1
            if patience_counter >= config['train']['early_stop_patience']:
                print("Early stopping triggered")
                break
                
    # Save Metrics
    with open(os.path.join(config['train']['save_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2)
        
    print("Training Complete.")

if __name__ == '__main__':
    main()
