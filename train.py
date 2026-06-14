import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm
from dataset import TokenizedPropertyDataset, collate_fn
from models import MultimodalXAIModel
from utils import load_config, set_seed, ensure_dirs, multitask_loss


def move_batch(batch, device):
    batch['numeric'] = batch['numeric'].to(device)
    batch['input_ids'] = batch['input_ids'].to(device)
    batch['attention_mask'] = batch['attention_mask'].to(device)
    batch['image'] = batch['image'].to(device)
    batch['graph'] = batch['graph'].to(device)
    for k in batch['labels']:
        batch['labels'][k] = batch['labels'][k].to(device)
    return batch


def main(config):
    cfg = load_config(config)
    set_seed(cfg['seed'])
    ensure_dirs(cfg['results_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    full = TokenizedPropertyDataset(cfg['data_dir'], tokenizer=tokenizer, image_size=cfg['image_size'])
    labels = full.df['suitability'].values
    train_idx, test_idx = train_test_split(np.arange(len(full)), test_size=1-cfg['train_ratio'], random_state=cfg['seed'], stratify=labels)

    train_ds = TokenizedPropertyDataset(cfg['data_dir'], train_idx, cfg['image_size'], tokenizer)
    test_ds = TokenizedPropertyDataset(cfg['data_dir'], test_idx, cfg['image_size'], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=cfg['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=cfg['num_workers'])

    model = MultimodalXAIModel(numeric_dim=len(full.feature_cols), dropout=cfg['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    history = []
    for epoch in range(cfg['epochs']):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg["epochs"]}'):
            batch = move_batch(batch, device)
            opt.zero_grad()
            outputs = model(batch)
            loss = multitask_loss(outputs, batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses))
        history.append({'epoch': epoch + 1, 'train_loss': mean_loss})
        print(f'Epoch {epoch+1}: train_loss={mean_loss:.4f}')

    torch.save({'model_state': model.state_dict(), 'feature_cols': full.feature_cols, 'test_idx': test_idx.tolist()}, os.path.join(cfg['results_dir'], 'multimodal_model.pt'))
    with open(os.path.join(cfg['results_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print('Saved model to results/multimodal_model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    main(args.config)
