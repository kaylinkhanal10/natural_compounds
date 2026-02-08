import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, edge_dim=4):
        """
        GNN Encoder for Molecules.
        
        Args:
            hidden_dim: Size of node/graph embeddings.
            num_layers: Number of GNN message passing layers.
            edge_dim: Dimension of edge attributes (bond types).
        """
        super(GNNEncoder, self).__init__()
        
        # Node embedding (Atom types, etc. - simplified for now, usually needs strict featurization)
        # Assuming input x has shape [num_nodes, num_node_features]
        # Let's assume initially we project raw features to hidden_dim
        # We need to know specific node feature dim. Let's assume 9 (common RDKit set)
        self.node_encoder = nn.Linear(9, hidden_dim) 
        
        # Edge embedding (Bond types)
        # Assuming edge_attr has shape [num_edges, edge_dim]
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # GINEConv requires an MLP for the update function
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            
        self.lns = nn.ModuleList()
        for _ in range(num_layers):
            self.lns.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: Node features [N, NodeFeatures]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, EdgeFeatures]
            batch: Batch vector [N] mapping nodes to graphs
        """
        
        # 1. Initial Embedding
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # 2. Message Passing
        for conv, ln in zip(self.convs, self.lns):
            x_res = x
            # GINEConv expects edge_attr to match node dim if we passed plain size
            # But GINEConv internals handle concatenation? 
            # PyG GINEConv: "The edge features are first projected by a linear layer..." NO.
            # GINEConv(nn, eps=0, train_eps=False, edge_dim=None, ...)
            # We must project edge_attr BEFORE if GINEConv doesn't do it?
            # Actually GINEConv takes `edge_attr` and adds it to node features before MLP.
            # So edge_attr must be [E, hidden_dim] match x [N, hidden_dim].
            # YES. We did ensure self.edge_encoder outputs hidden_dim.
            
            x = conv(x, edge_index, edge_attr)
            x = ln(x)
            x = F.relu(x)
            x = x + x_res # Residual connection (optional but good)
            
        # 3. Global Pooling (Readout)
        # Sum pooling is generally good for counting constructs (atoms)
        # Add pooling is standard for molecule tasks
        graph_emb = global_add_pool(x, batch)
        
        return graph_emb
