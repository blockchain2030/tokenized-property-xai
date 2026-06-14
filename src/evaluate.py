import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import TokenizedPropertyDataset, collate_fn
from models import MultimodalXAIModel
from train import move_batch
from utils import load_config, evaluate_multitask_predictions


def main(config):
    cfg = load_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(os.path.join(cfg['results_dir'], 'multimodal_model.pt'), map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    ds = TokenizedPropertyDataset(cfg['data_dir'], ckpt['test_idx'], cfg['image_size'], tokenizer)
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = MultimodalXAIModel(numeric_dim=len(ckpt['feature_cols']), dropout=cfg['dropout']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    y_true = {'suitability': [], 'appreciation': [], 'liquidity_risk': []}
    y_pred = {'suitability': [], 'appreciation': [], 'liquidity_risk': []}
    y_prob = {'suitability': [], 'appreciation': [], 'liquidity_risk': []}

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch)
            for task in y_true:
                prob = torch.softmax(out[task], dim=1).cpu().numpy()
                pred = prob.argmax(axis=1)
                y_prob[task].append(prob)
                y_pred[task].extend(pred.tolist())
                y_true[task].extend(batch['labels'][task].cpu().numpy().tolist())

    y_prob = {k: np.vstack(v) for k, v in y_prob.items()}
    metrics = evaluate_multitask_predictions(y_true, y_pred, y_prob)
    with open(os.path.join(cfg['results_dir'], 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    main(args.config)
