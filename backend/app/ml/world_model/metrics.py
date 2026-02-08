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

def compute_retrieval_metrics(y_true, y_pred_logits, k=10):
    """
    Computes Retrieval@K (Recall@K) for multi-label classification.
    y_true: [batch, num_targets] (0 or 1)
    y_pred_logits: [batch, num_targets] (logits)
    """
    # Get top k indices
    _, topk_indices = torch.topk(y_pred_logits, k=k, dim=1)
    
    # Check if any true label is in top k
    # Gather true labels at topk indices
    # shape: [batch, k]
    batch_size = y_true.size(0)
    
    hits = 0
    for i in range(batch_size):
        # Indices of true targets
        true_indices = (y_true[i] == 1).nonzero(as_tuple=True)[0]
        if len(true_indices) == 0:
            continue
            
        # Check intersection
        pred_indices = topk_indices[i]
        
        # Intersection
        # We want to know if AT LEAST ONE true target was retrieved?
        # Or Recall@K (fraction of true targets retrieved)?
        # Usually for "Retrieval@K" in this context we mean "Hit@K" (is the relevant item there?)
        # Since we have multiple targets, let's use Recall@K averaged
        
        retrieved_count = 0
        for idx in pred_indices:
            if idx in true_indices:
                retrieved_count += 1
        
        recall = retrieved_count / len(true_indices)
        hits += recall
        
    avg_recall = hits / batch_size
    return {'recall_at_k': avg_recall}
