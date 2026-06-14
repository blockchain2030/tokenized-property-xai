import torch
from torch import nn
from .numerical_mlp import NumericalMLPEncoder
from .text_bert import BERTTextEncoder
from .image_resnet50 import ResNet50ImageEncoder
from .blockchain_gat import BlockchainGATEncoder
from .cross_attention_fusion import CrossAttentionFusion

class MultimodalXAIModel(nn.Module):
    """Full manuscript-aligned model: MLP + BERT + ResNet50 + GAT + cross-attention + 3 heads."""
    def __init__(self, numerical_dim: int, graph_node_dim: int):
        super().__init__()
        self.num = NumericalMLPEncoder(numerical_dim)
        self.txt = BERTTextEncoder()
        self.img = ResNet50ImageEncoder()
        self.gph = BlockchainGATEncoder(graph_node_dim)
        self.fusion = CrossAttentionFusion()
        self.shared = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.suitability = nn.Linear(256, 1)
        self.capital_appreciation = nn.Linear(256, 3)
        self.liquidity_risk = nn.Linear(256, 3)
    def forward(self, batch):
        z_num = self.num(batch['numerical'])
        z_txt = self.txt(batch['input_ids'], batch['attention_mask'])
        z_img = self.img(batch['images'])
        z_gph = self.gph(batch['graph_x'], batch['edge_index'], batch['graph_batch'])
        fused, attn = self.fusion(z_num, z_txt, z_img, z_gph)
        h = self.shared(fused)
        return {
            'suitability': self.suitability(h).squeeze(-1),
            'capital_appreciation': self.capital_appreciation(h),
            'liquidity_risk': self.liquidity_risk(h),
            'cross_attention': attn
        }
