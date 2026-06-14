import torch
from torch import nn

class CrossAttentionFusion(nn.Module):
    """Projects four modality embeddings into 256D and applies 4-head cross-attention."""
    def __init__(self):
        super().__init__()
        self.proj_num = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256))
        self.proj_txt = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256))
        self.proj_img = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256))
        self.proj_gph = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256))
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
        self.norm = nn.LayerNorm(256)
    def forward(self, z_num, z_txt, z_img, z_gph):
        z = torch.stack([
            self.proj_num(z_num), self.proj_txt(z_txt), self.proj_img(z_img), self.proj_gph(z_gph)
        ], dim=1)
        attended, weights = self.attn(z, z, z, need_weights=True)
        fused = self.norm(attended + self.ffn(attended))
        return fused.mean(dim=1), weights
