import torch
import yaml
import os
import json
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .dataset import MMVAEDataset
from .model import MultiModalVAE
from .losses import VAELoss
from .metrics import compute_chemical_metrics, check_validity

def train_one_epoch(model, dataloader, criterion, optimizer, beta, device):
    model.train()
    total_loss = 0
    total_chem = 0
    total_kld = 0
    
    for x_chem in dataloader:
        x_chem = x_chem.to(device)
        
        optimizer.zero_grad()
        recon_chem, mu, logvar, z = model(x_chem)
        
        loss, l_chem, kld = criterion(recon_chem, x_chem, mu, logvar, beta)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_chem += l_chem.item()
        total_kld += kld.item()
            
    avg_loss = total_loss / len(dataloader)
    
    metrics = {
        'train_total_loss': avg_loss,
        'train_recon_chem': total_chem / len(dataloader),
        'train_kld': total_kld / len(dataloader),
    }
    return metrics

def validate(model, dataloader, criterion, beta, device):
    model.eval()
    total_loss = 0
    total_chem = 0
    total_kld = 0
    
    # Accumulate for full-set metrics
    all_real_chem = []
    all_pred_chem = []
    
    with torch.no_grad():
        for x_chem in dataloader:
            x_chem = x_chem.to(device)
            
            recon_chem, mu, logvar, z = model(x_chem)
            loss, l_chem, kld = criterion(recon_chem, x_chem, mu, logvar, beta)
            
            total_loss += loss.item()
            total_chem += l_chem.item()
            total_kld += kld.item()
            
            # Store for batch-independent metrics
            all_real_chem.append(x_chem)
            all_pred_chem.append(recon_chem)
            
    # Concatenate
    real_c = torch.cat(all_real_chem, dim=0)
    pred_c = torch.cat(all_pred_chem, dim=0)
    
    # Compute Advanced Metrics
    chem_metrics = compute_chemical_metrics(real_c, pred_c)
    validity = check_validity(pred_c)
            
    metrics = {
        'val_total_loss': total_loss / len(dataloader),
        'val_recon_chem': total_chem / len(dataloader),
        'val_kld': total_kld / len(dataloader),
        **chem_metrics,
        **validity
    }
    return metrics

def plot_metrics(history, save_dir):
    """Generate and save analysis plots from metrics history."""
    df = pd.DataFrame(history)
    os.makedirs(os.path.join(save_dir, 'figures'), exist_ok=True)
    
    # 1. Losses
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_total_loss'], label='Train Total')
    plt.plot(df['epoch'], df['val_total_loss'], label='Val Total')
    plt.title('Total Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'figures', 'loss_curve.png'))
    plt.close()
    
    # 2. Chemical Quality (R2)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['test_chem_r2'], color='green', label='R2 Score')
    plt.title('Chemical Reconstruction Quality (R2)')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'figures', 'chem_r2.png'))
    plt.close()
    

    
    # 4. KLD
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_kld'], color='orange', label='Train KLD')
    plt.title('KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('KLD')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'figures', 'kld.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='backend/app/ml/world_model/config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['train']['save_dir'], exist_ok=True)
    device = torch.device('cpu') # Forced
    print(f"Using device: {device}")
    
    # 1. Dataset
    dataset = MMVAEDataset(data_dir=config['data']['raw_dir'], config_path=args.config, split='train')
    val_dataset = MMVAEDataset(data_dir=config['data']['raw_dir'], config_path=args.config, split='val')
    
    train_loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    chem_dim = dataset.X_chem.shape[1]
    print(f"Chem Dim: {chem_dim}")
    
    # 2. Model
    # 2. Model
    model = MultiModalVAE(chem_dim=chem_dim, latent_dim=config['model']['latent_dim']).to(device)
    
    # 3. Optimizer & Loss
    # 3. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    criterion = VAELoss(lambda_chem=config['train']['lambda_chem']).to(device)
    
    epochs = config['train']['epochs']
    warmup = config['train']['beta_warmup_epochs']
    
    metrics_history = []
    
    for epoch in range(epochs):
        if epoch < warmup:
            beta = epoch / warmup
        else:
            beta = 1.0
            
        current_epoch = epoch + 1
            
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, beta, device)
        val_metrics = validate(model, val_loader, criterion, beta, device)
        
        # Merge metrics
        epoch_log = {'epoch': current_epoch, 'beta': beta}
        epoch_log.update(train_metrics)
        epoch_log.update(val_metrics)
        
        print(f"Epoch {current_epoch}/{epochs} | "
              f"L_Tot: {train_metrics['train_total_loss']:.4f} | "
              f"R2: {val_metrics['test_chem_r2']:.4f}")
        
        metrics_history.append(epoch_log)
        
        # Save every epoch to ensure logging persistence
        with open(os.path.join(config['train']['save_dir'], 'metrics.json'), 'w') as f:
            json.dump(metrics_history, f, indent=2)
            
        # Stopping Logic based on R2 Stability (New robustness check)
        if val_metrics['test_chem_r2'] > 0.99: 
             # Early stop if perfect reconstruction (unlikely but possible)
             pass
            
        torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], 'final_model.pt'))
                
    # Finalize
    # Save CSV
    df = pd.DataFrame(metrics_history)
    df.to_csv(os.path.join(config['train']['save_dir'], 'metrics.csv'), index=False)
    
    # Generate Plots
    print("Generating plots...")
    plot_metrics(metrics_history, config['train']['save_dir'])
    print("Training Complete. Metrics and Figures saved.")

if __name__ == '__main__':
    main()
