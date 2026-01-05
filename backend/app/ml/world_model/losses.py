import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, lambda_chem=1.0):
        super(VAELoss, self).__init__()
        self.lambda_chem = lambda_chem
        self.mse_loss = nn.MSELoss()

    def forward(self, recon_chem, x_chem, mu, logvar, beta=1.0):
        # 1. Chemical Reconstruction (MSE)
        loss_chem = self.mse_loss(recon_chem, x_chem)
        
        # 2. KL Divergence
        # KL(N(mu, sigma), N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalize KLD by batch size
        kld = kld / x_chem.size(0)
        
        # Total Loss
        total_loss = (self.lambda_chem * loss_chem) + \
                     (beta * kld)
                     
        return total_loss, loss_chem, kld
