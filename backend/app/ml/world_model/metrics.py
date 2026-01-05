import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, precision_recall_curve, auc, f1_score, jaccard_score

def compute_chemical_metrics(real, pred):
    """
    Computes regression metrics for chemical reconstruction.
    real, pred: torch tensors (batch, dim)
    Returns dict: r2, nrmse (normalized RMSE)
    """
    real_np = real.cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    
    # R2 Score
    # Handle occasional batch size of 1 or constant values which warn in sklearn
    try:
        r2 = r2_score(real_np, pred_np)
    except:
        r2 = 0.0
        
    # NRMSE (RMSE / std of real data)
    mse = mean_squared_error(real_np, pred_np)
    rmse = np.sqrt(mse)
    std = np.std(real_np)
    nrmse = rmse / (std + 1e-9)
    
    return {
        'test_chem_r2': float(r2),
        'test_chem_nrmse': float(nrmse)
    }

def compute_protein_metrics(real_binary, pred_logits, threshold=0.5):
    """
    Computes multi-label classification metrics.
    real_binary: Ground truth binary matrix (batch, dim)
    pred_logits: Raw logits from model (batch, dim)
    """
    real_np = real_binary.cpu().numpy()
    # Apply sigmoid
    probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
    preds = (probs > threshold).astype(int)
    
    # PR-AUC (Micro-averaged for multi-label)
    # Flatten arrays for micro-average
    try:
        precision, recall, _ = precision_recall_curve(real_np.ravel(), probs.ravel())
        pr_auc = auc(recall, precision)
    except:
        pr_auc = 0.0
        
    # F1 Score (Micro)
    f1 = f1_score(real_np, preds, average='micro', zero_division=0)
    
    # Jaccard Score (Samples average)
    jaccard = jaccard_score(real_np, preds, average='samples', zero_division=0)
    
    # Activation Stats
    # Mean predicted positives per sample
    mean_pred_count = np.mean(np.sum(preds, axis=1))
    median_pred_count = np.median(np.sum(preds, axis=1))
    
    return {
        'test_prot_prauc': float(pr_auc),
        'test_prot_f1_micro': float(f1),
        'test_prot_jaccard': float(jaccard),
        'test_pred_count_mean': float(mean_pred_count),
        'test_pred_count_median': float(median_pred_count)
    }

def check_validity(chem_recon, prot_recon_logits, chem_bounds=None):
    """
    Checks if reconstructions are within "sane" bounds.
    """
    # 1. Chemical Validity (No NaN/Inf)
    has_nan = torch.isnan(chem_recon).any().item()
    has_inf = torch.isinf(chem_recon).any().item()
    
    valid_chem = not (has_nan or has_inf)
    
    # 2. Protein Validity
    # Check if predicted counts are not zero (collapse) or all ones (explosion)
    probs = torch.sigmoid(prot_recon_logits)
    preds = (probs > 0.5).float()
    counts = preds.sum(dim=1)
    
    # Arbitrary validity: at least 1 target, less than 500 (heuristic)
    valid_prot_mask = (counts >= 1) & (counts < 500)
    valid_prot_pct = valid_prot_mask.float().mean().item()
    
    return {
        'valid_chem': 1.0 if valid_chem else 0.0,
        'valid_prot_pct': valid_prot_pct
    }
