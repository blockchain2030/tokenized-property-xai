import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import AutoModel
from torch_geometric.nn import GATConv, global_mean_pool


class NumericEncoder(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', out_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.proj(pooled)


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Linear(base.fc.in_features, out_dim)
    def forward(self, image):
        feat = self.features(image).flatten(1)
        return self.proj(feat)


class GraphEncoder(nn.Module):
    def __init__(self, in_channels=6, hidden=128, out_dim=256, heads=8):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden, heads=heads, concat=True)
        self.gat2 = GATConv(hidden * heads, out_dim, heads=1, concat=False)
        self.attention_weights = None
    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = torch.relu(self.gat1(x, edge_index))
        x, att = self.gat2(x, edge_index, return_attention_weights=True)
        self.attention_weights = att
        return global_mean_pool(torch.relu(x), batch)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.num_proj = nn.Linear(128, 256)
        self.text_proj = nn.Linear(256, 256)
        self.img_proj = nn.Linear(512, 256)
        self.graph_proj = nn.Linear(256, 256)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256))
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
    def forward(self, n, t, i, g):
        tokens = torch.stack([self.num_proj(n), self.text_proj(t), self.img_proj(i), self.graph_proj(g)], dim=1)
        attn_out, weights = self.attn(tokens, tokens, tokens, need_weights=True)
        x = self.norm1(tokens + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1), weights


class MultimodalXAIModel(nn.Module):
    def __init__(self, numeric_dim, dropout=0.3):
        super().__init__()
        self.numeric_encoder = NumericEncoder(numeric_dim)
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.graph_encoder = GraphEncoder()
        self.fusion = CrossAttentionFusion(dropout)
        self.shared = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 256), nn.ReLU())
        self.suitability = nn.Linear(256, 2)
        self.appreciation = nn.Linear(256, 3)
        self.liquidity_risk = nn.Linear(256, 3)
        self.last_fusion_attention = None
    def forward(self, batch):
        n = self.numeric_encoder(batch['numeric'])
        t = self.text_encoder(batch['input_ids'], batch['attention_mask'])
        i = self.image_encoder(batch['image'])
        g = self.graph_encoder(batch['graph'])
        fused, weights = self.fusion(n, t, i, g)
        self.last_fusion_attention = weights
        h = self.shared(fused)
        return {
            'suitability': self.suitability(h),
            'appreciation': self.appreciation(h),
            'liquidity_risk': self.liquidity_risk(h),
        }
