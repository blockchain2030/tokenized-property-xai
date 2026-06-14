import torch
from torch import nn

try:
    from torch_geometric.nn import GATConv, global_mean_pool
except Exception:
    GATConv = None
    global_mean_pool = None

class BlockchainGATEncoder(nn.Module):
    """Two-layer GAT with 8 attention heads, matching manuscript design."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, heads: int = 8, out_dim: int = 256):
        super().__init__()
        if GATConv is None:
            raise ImportError('Install torch-geometric to use BlockchainGATEncoder.')
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=0.2)
        self.attention_weights = None
    def forward(self, x, edge_index, batch):
        x, att1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = torch.relu(x)
        x, att2 = self.gat2(x, edge_index, return_attention_weights=True)
        self.attention_weights = {'layer1': att1, 'layer2': att2}
        return global_mean_pool(x, batch)
