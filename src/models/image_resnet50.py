import torch
from torch import nn
from torchvision import models

class ResNet50ImageEncoder(nn.Module):
    """ResNet-50 ImageNet backbone with 512-dimensional projection."""
    def __init__(self, out_dim: int = 512):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        base = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Linear(base.fc.in_features, out_dim)
        self.target_layer = base.layer4
    def forward(self, images):
        x = self.features(images).flatten(1)
        return self.proj(x)
