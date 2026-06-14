import torch
from torch import nn

def train_one_epoch(model, loader, optimizer, device='cuda'):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        out = model(batch)
        loss = (
            bce(out['suitability'], batch['y_suitability'].float()) +
            ce(out['capital_appreciation'], batch['y_capital']) +
            ce(out['liquidity_risk'], batch['y_liquidity'])
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)
