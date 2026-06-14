"""Generate a synthetic/demo dataset matching the paper structure.
Replace this with real tokenized-property data for final experiments.
"""
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch_geometric.data import Data
from utils import ensure_dirs, set_seed, load_config


def generate(config_path='configs/config.yaml'):
    cfg = load_config(config_path)
    set_seed(cfg['seed'])
    data_dir = cfg['data_dir']
    ensure_dirs(data_dir, f'{data_dir}/images', f'{data_dir}/graphs')

    n = cfg['num_samples']
    ids = [f'INV_{i:04d}' for i in range(n)]

    numeric_rows = []
    text_rows = []
    label_rows = []

    legal_positive = ['clear title', 'low restriction', 'stable rental income', 'audited disclosure']
    legal_negative = ['penalty clause', 'transfer restriction', 'high volatility', 'ownership concentration', 'liquidity warning']

    for i, inv_id in enumerate(ids):
        rental_yield = np.random.normal(0.07, 0.02)
        volatility = abs(np.random.normal(0.25, 0.10))
        liquidity_depth = abs(np.random.normal(0.55, 0.20))
        token_concentration = np.clip(np.random.beta(2, 5), 0, 1)
        trading_volume = abs(np.random.normal(100000, 40000))
        market_cap = abs(np.random.normal(2000000, 700000))
        hhi = np.clip(token_concentration + np.random.normal(0, 0.08), 0, 1)
        pagerank_score = np.clip(np.random.beta(2, 6), 0, 1)

        risk_score = (volatility * 1.8) + (token_concentration * 1.5) + (hhi * 1.2) - (rental_yield * 4.0) - (liquidity_depth * 0.8)
        suitability = int(risk_score < 0.75)
        appreciation = 2 if (rental_yield > 0.075 and volatility < 0.25) else (1 if rental_yield > 0.055 else 0)
        liquidity_risk = 2 if liquidity_depth < 0.35 or token_concentration > 0.55 else (1 if liquidity_depth < 0.55 else 0)

        numeric_rows.append({
            'investment_id': inv_id,
            'rental_yield': rental_yield,
            'price_volatility': volatility,
            'liquidity_depth': liquidity_depth,
            'token_concentration': token_concentration,
            'trading_volume': trading_volume,
            'market_cap': market_cap,
            'hhi': hhi,
            'pagerank_score': pagerank_score,
        })

        words = legal_positive if suitability else legal_negative
        text = ' '.join(np.random.choice(words, size=12, replace=True))
        text_rows.append({'investment_id': inv_id, 'document_text': text})
        label_rows.append({'investment_id': inv_id, 'suitability': suitability, 'appreciation': appreciation, 'liquidity_risk': liquidity_risk})

        for j in range(4):
            img = Image.new('RGB', (224, 224), color=(230, 230, 230))
            d = ImageDraw.Draw(img)
            shade = int(80 + 100 * suitability + np.random.randint(-20, 20))
            d.rectangle([40, 60, 180, 190], outline=(shade, shade, shade), width=4)
            d.rectangle([65, 85, 100, 125], outline=(120, 120, 120), width=2)
            d.rectangle([125, 85, 160, 125], outline=(120, 120, 120), width=2)
            d.line([40, 60, 110, 25, 180, 60], fill=(shade, shade, shade), width=4)
            img.save(f'{data_dir}/images/{inv_id}_{j}.jpg')

        num_nodes = np.random.randint(8, 20)
        x = torch.randn(num_nodes, 6)
        edge_count = np.random.randint(num_nodes, num_nodes * 3)
        edge_index = torch.randint(0, num_nodes, (2, edge_count), dtype=torch.long)
        edge_attr = torch.rand(edge_count, 3)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        torch.save(graph, f'{data_dir}/graphs/{inv_id}.pt')

    pd.DataFrame(numeric_rows).to_csv(f'{data_dir}/numeric.csv', index=False)
    pd.DataFrame(text_rows).to_csv(f'{data_dir}/text.csv', index=False)
    pd.DataFrame(label_rows).to_csv(f'{data_dir}/labels.csv', index=False)
    print(f'Generated demo dataset with {n} investment instances in {data_dir}/')


if __name__ == '__main__':
    generate()
