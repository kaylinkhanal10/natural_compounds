import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class MoleculeEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, out_dim=128, num_layers=3):
        super(MoleculeEncoder, self).__init__()
        
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # GINEConv requires an MLP for the update function
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        # x: [num_nodes, node_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        # batch: [num_nodes] (batch index for each node)

        x = self.node_embedding(x)
        original_edge_attr = self.edge_embedding(edge_attr)
        
        for i, conv in enumerate(self.convs):
            # GINEConv expects edge_attr to be same dim as node features
            x = conv(x, edge_index, original_edge_attr) 
            x = self.bn_layers[i](x)
            x = F.relu(x)
            
        if batch is None:
            # Assume single graph if no batch provided
             batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Global Pooling
        x = global_mean_pool(x, batch)
        
        # Final Projection
        emb = self.readout(x)
        
        return emb
