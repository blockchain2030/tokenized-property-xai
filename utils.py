import os
import random
import yaml
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def multitask_loss(outputs, targets):
    ce = torch.nn.CrossEntropyLoss()
    suitability_loss = ce(outputs['suitability'], targets['suitability'])
    appreciation_loss = ce(outputs['appreciation'], targets['appreciation'])
    liquidity_loss = ce(outputs['liquidity_risk'], targets['liquidity_risk'])
    return suitability_loss + appreciation_loss + liquidity_loss


def evaluate_multitask_predictions(y_true, y_pred, y_prob):
    metrics = {}
    for task in ['suitability', 'appreciation', 'liquidity_risk']:
        metrics[f'{task}_accuracy'] = accuracy_score(y_true[task], y_pred[task])
        metrics[f'{task}_f1_macro'] = f1_score(y_true[task], y_pred[task], average='macro')
        try:
            if len(np.unique(y_true[task])) == 2:
                metrics[f'{task}_roc_auc'] = roc_auc_score(y_true[task], y_prob[task][:, 1])
            else:
                metrics[f'{task}_roc_auc'] = roc_auc_score(y_true[task], y_prob[task], multi_class='ovr')
        except Exception:
            metrics[f'{task}_roc_auc'] = np.nan
    metrics['weighted_accuracy'] = np.mean([metrics[f'{t}_accuracy'] for t in ['suitability','appreciation','liquidity_risk']])
    metrics['weighted_f1'] = np.mean([metrics[f'{t}_f1_macro'] for t in ['suitability','appreciation','liquidity_risk']])
    metrics['weighted_roc_auc'] = np.nanmean([metrics[f'{t}_roc_auc'] for t in ['suitability','appreciation','liquidity_risk']])
    return metrics
