import torch
import torch.nn as nn

class PlausibilityDeepSet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        """
        DeepSets architecture for Set-based Plausibility Scoring.
        Score = Rho( Sum( Phi(x) ) )
        """
        super().__init__()
        
        # Phi: Process each element individually
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Rho: Process the aggregated set
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Probability 0-1
        )
        
    def forward(self, x, batch=None):
        """
        Args:
            x: [Total_Compounds, Dim] - Stacked embeddings of all compounds in the batch
            batch: [Total_Compounds] - Index tensor indicating which set each compound belongs to
                   (Standard PyG batching format)
        Return:
            score: [Num_Sets, 1]
        """
        
        # 1. Transform elements (Phi)
        h = self.phi(x)
        
        # 2. Aggregate (Sum Pooling)
        # If batch is None, assume single set input [N, D] -> sum -> [1, D]
        if batch is None:
            h_sum = torch.sum(h, dim=0, keepdim=True)
        else:
            # Global pooling using scatter add (or simple loop/pandas, but torch_scatter is best)
            # Since we don't know if torch_scatter is installed, we use a simple loop or index_add
            # For efficiency in standard PyTorch:
            num_sets = batch.max().item() + 1
            h_sum = torch.zeros(num_sets, h.size(1), device=x.device)
            h_sum.index_add_(0, batch, h)
            
        # 3. Predict Set Property (Rho)
        score = self.rho(h_sum)
        
        return score
