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

def check_validity(chem_recon):
    """
    Checks if reconstructions are within "sane" bounds.
    """
    # 1. Chemical Validity (No NaN/Inf)
    has_nan = torch.isnan(chem_recon).any().item()
    has_inf = torch.isinf(chem_recon).any().item()
    
    valid_chem = not (has_nan or has_inf)
    
    return {
        'valid_chem': 1.0 if valid_chem else 0.0
    }
