import torch
from torch import nn
from transformers import AutoModel

class BERTTextEncoder(nn.Module):
    """BERT text encoder with mean pooling and 256-dimensional projection."""
    def __init__(self, model_name: str = 'bert-base-uncased', out_dim: int = 256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.proj(pooled)
