import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

@torch.no_grad()
def evaluate_binary_suitability(model, loader, device='cuda'):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        out = model(batch)
        prob = torch.sigmoid(out['suitability']).detach().cpu().numpy()
        y_prob.extend(prob.tolist())
        y_true.extend(batch['y_suitability'].detach().cpu().numpy().tolist())
    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }
