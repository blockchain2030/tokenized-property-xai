import torch
from torch import nn

class NumericalMLPEncoder(nn.Module):
    """Numerical encoder matching manuscript: 256 -> 128 -> 128 with BatchNorm and Dropout."""
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)
