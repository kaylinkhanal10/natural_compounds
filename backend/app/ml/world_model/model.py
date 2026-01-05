import torch
import torch.nn as nn
import torch.nn.functional as F

class ChemEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ChemEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)



class MultiModalVAE(nn.Module):
    def __init__(self, chem_dim, latent_dim=32):
        super(MultiModalVAE, self).__init__()
        
        self.chem_encoder = ChemEncoder(chem_dim)
        
        # Latent Distribution (from chem hidden directly)
        # ChemEncoder outputs 128
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        
        # Chem Decoder
        self.chem_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, chem_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_chem):
        # Encode
        h_chem = self.chem_encoder(x_chem)
        
        # Latent space
        mu = self.fc_mu(h_chem)
        logvar = self.fc_var(h_chem)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_chem = self.chem_decoder(z)
        
        return recon_chem, mu, logvar, z
