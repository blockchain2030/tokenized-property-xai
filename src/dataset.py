import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch_geometric.data import Batch


class TokenizedPropertyDataset(Dataset):
    def __init__(self, data_dir, indices=None, image_size=224, tokenizer=None):
        self.data_dir = data_dir
        self.numeric = pd.read_csv(os.path.join(data_dir, 'numeric.csv'))
        self.text = pd.read_csv(os.path.join(data_dir, 'text.csv'))
        self.labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
        self.df = self.numeric.merge(self.text, on='investment_id').merge(self.labels, on='investment_id')
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.feature_cols = [c for c in self.numeric.columns if c != 'investment_id']
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inv_id = row['investment_id']
        numeric = torch.tensor(row[self.feature_cols].values.astype('float32'))

        text = str(row['document_text'])
        if self.tokenizer:
            toks = self.tokenizer(text, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
            input_ids = toks['input_ids'].squeeze(0)
            attention_mask = toks['attention_mask'].squeeze(0)
        else:
            input_ids = torch.zeros(64, dtype=torch.long)
            attention_mask = torch.zeros(64, dtype=torch.long)

        image_path = os.path.join(self.data_dir, 'images', f'{inv_id}_0.jpg')
        image = self.transform(Image.open(image_path).convert('RGB'))
        graph = torch.load(os.path.join(self.data_dir, 'graphs', f'{inv_id}.pt'), weights_only=False)

        labels = {
            'suitability': torch.tensor(int(row['suitability']), dtype=torch.long),
            'appreciation': torch.tensor(int(row['appreciation']), dtype=torch.long),
            'liquidity_risk': torch.tensor(int(row['liquidity_risk']), dtype=torch.long),
        }
        return {
            'investment_id': inv_id,
            'numeric': numeric,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image,
            'graph': graph,
            'labels': labels,
            'text': text,
        }


def collate_fn(batch):
    return {
        'investment_id': [b['investment_id'] for b in batch],
        'numeric': torch.stack([b['numeric'] for b in batch]),
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'image': torch.stack([b['image'] for b in batch]),
        'graph': Batch.from_data_list([b['graph'] for b in batch]),
        'labels': {
            'suitability': torch.stack([b['labels']['suitability'] for b in batch]),
            'appreciation': torch.stack([b['labels']['appreciation'] for b in batch]),
            'liquidity_risk': torch.stack([b['labels']['liquidity_risk'] for b in batch]),
        },
        'text': [b['text'] for b in batch],
    }
