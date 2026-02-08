import torch
import torch.nn as nn

class TargetPredictionHead(nn.Module):
    def __init__(self, input_dim, num_targets, hidden_dim=256):
        super(TargetPredictionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_targets)
            # No Sigmoid here, use BCEWithLogitsLoss
        )
        
    def forward(self, x):
        return self.net(x)

class ActivityRegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ActivityRegressionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)
