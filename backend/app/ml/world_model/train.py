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
    mean_activations = []
    
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
        
        # Diagnostics
        with torch.no_grad():
             # Sigmoid to get probabilities
             probs = torch.sigmoid(recon_prot)
             mean_activations.append(probs.mean().item())
        
    avg_loss = total_loss / len(dataloader)
    avg_act = sum(mean_activations) / len(mean_activations)
    return avg_loss, total_chem/len(dataloader), total_prot/len(dataloader), total_kld/len(dataloader), avg_act

def validate(model, dataloader, criterion, beta, device):
    model.eval()
    total_loss = 0
    total_chem = 0
    total_prot = 0
    total_kld = 0
    mean_activations = []
    
    with torch.no_grad():
        for x_chem, x_prot in dataloader:
            x_chem = x_chem.to(device)
            x_prot = x_prot.to(device)
            
            recon_chem, recon_prot, mu, logvar, z = model(x_chem, x_prot)
            loss, l_chem, l_prot, kld = criterion(recon_chem, x_chem, recon_prot, x_prot, mu, logvar, beta)
            
            # Diagnostics
            with torch.no_grad():
                probs = torch.sigmoid(recon_prot)
                mean_activations.append(probs.mean().item())
            
            total_loss += loss.item()
            total_chem += l_chem.item()
            total_prot += l_prot.item()
            total_kld += kld.item()
            
    avg_loss = total_loss / len(dataloader)
    avg_act = sum(mean_activations) / len(mean_activations) if mean_activations else 0.0
    return avg_loss, total_chem/len(dataloader), total_prot/len(dataloader), total_kld/len(dataloader), avg_act

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='backend/app/ml/world_model/config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(config['train']['save_dir'], exist_ok=True)
    os.makedirs(config['train']['save_dir'], exist_ok=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu') # Forced due to RTX 5090 CUDA incompatibility
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
    # FIXED: Hardcoded to 20.0 as per stability repair instructions
    pos_weight = 20.0
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
    collapse_counter = 0
    epochs = config['train']['epochs']
    warmup = config['train']['beta_warmup_epochs']
    
    metrics_history = []
    
    for epoch in range(epochs):
        # Linear Beta Warmup
        if epoch < warmup:
            beta = epoch / warmup
        else:
            beta = 1.0
            
        # Dynamic Loss Scheduling
        # lambda_prot = max(0.4, 1.0 - (epoch / 50.0))
        # epoch is 0-indexed in code, so use epoch+1 for 1-based formula or align as needed.
        # User specified "epoch is 1-indexed" in formula.
        current_epoch = epoch + 1
        lambda_prot = max(0.4, 1.0 - (current_epoch / 50.0))
        criterion.lambda_prot = lambda_prot
            
        # Updated unpacking
        train_loss, t_chem, t_prot, t_kld, t_act = train_one_epoch(model, train_loader, criterion, optimizer, beta, device)
        val_loss, v_chem, v_prot, v_kld, v_act = validate(model, val_loader, criterion, beta, device)
        
        print(f"Epoch {current_epoch}/{epochs} | Beta: {beta:.2f} | L_Prot: {lambda_prot:.2f} | Train: {train_loss:.4f} | Val: {val_loss:.4f} (Act: {v_act:.5f})")
        
        metrics_history.append({
            'epoch': current_epoch,
            'train_loss': train_loss, 'val_loss': val_loss,
            'val_chem': v_chem, 'val_prot': v_prot,
            'train_act': t_act, 'val_act': v_act,
            'lambda_prot': lambda_prot
        })
        
        # --- STOPPING LOGIC ---
        
        # 1. Track Activation History
        # We need last 3 epochs.
        recent_acts = [m['val_act'] for m in metrics_history[-3:]]
        
        # Condition A: Convergence (Desired)
        # val_act >= 0.001 AND stable (<10% change) for 3 epochs
        is_converged = False
        if len(recent_acts) == 3:
            # Check magnitude
            if all(a >= 0.001 for a in recent_acts):
                # Check stability: change < 10%
                # abs(a2-a1)/a1 < 0.1 AND abs(a3-a2)/a2 < 0.1
                chg1 = abs(recent_acts[1] - recent_acts[0]) / (recent_acts[0] + 1e-9)
                chg2 = abs(recent_acts[2] - recent_acts[1]) / (recent_acts[1] + 1e-9)
                
                if chg1 < 0.10 and chg2 < 0.10:
                    is_converged = True
                    
        if is_converged:
            print(f"SUCCESS: Protein signal stabilized (Act >= 0.001 and stable). Freezing checkpoint.")
            torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], 'best_model.pt'))
            break

        # Condition B: Collapse Prevention
        # val_act < 0.001 for 3 consecutive epochs
        is_collapsed = False
        if len(recent_acts) == 3:
             if all(a < 0.001 for a in recent_acts):
                 is_collapsed = True
                 
        if is_collapsed:
            print(f"CRITICAL: Protein signal collapsed (Act < 0.001 for 3 epochs). Stopping.")
            torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], 'best_model.pt'))
            break
            
        # Standard Save (Best Loss) - Optional backup
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], 'best_model.pt'))
            # User wants to freeze on STABILITY, not lowest loss. 
            # But we should save 'checkpoint.pt' maybe? 
            # User instruction: "Apply this change ... freeze the first stable checkpoint."
            # We will rely on Condition A to save the 'best' model relative to the goal.
            pass
                
    # Save Metrics
    with open(os.path.join(config['train']['save_dir'], 'metrics.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2)
        
    print("Training Complete.")

if __name__ == '__main__':
    main()
