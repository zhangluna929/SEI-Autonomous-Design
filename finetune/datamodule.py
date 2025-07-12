import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import torch
import numpy as np
from ..pretrain.datamodule import MultimodalDataset, collate_fn

class LabeledDataset(MultimodalDataset):
    def __init__(self, parquet_path: Path):
        super().__init__(parquet_path)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        row = self.df.iloc[idx]
        
        # 添加错误处理和默认值
        try:
            item['deltaE'] = torch.tensor(float(row.get('deltaE', 0.0)), dtype=torch.float)
            # 处理sigma的对数，避免负值或零值
            sigma_val = float(row.get('sigma', 1e-6))
            if sigma_val <= 0:
                sigma_val = 1e-6
            item['sigma'] = torch.tensor(np.log(sigma_val), dtype=torch.float)
            item['fermi'] = torch.tensor(float(row.get('fermi_pin', 0.0)), dtype=torch.float)
            item['temp'] = torch.tensor(float(row.get('temp', 298.0)), dtype=torch.float)
            item['synth_diff'] = torch.tensor(float(row.get('synthesis_difficulty', 0.5)), dtype=torch.float)
        except (ValueError, TypeError) as e:
            # 如果数据转换失败，使用默认值
            item['deltaE'] = torch.tensor(0.0, dtype=torch.float)
            item['sigma'] = torch.tensor(np.log(1e-6), dtype=torch.float)
            item['fermi'] = torch.tensor(0.0, dtype=torch.float)
            item['temp'] = torch.tensor(298.0, dtype=torch.float)
            item['synth_diff'] = torch.tensor(0.5, dtype=torch.float)
            
        return item

def collate_fn_label(batch):
    data = collate_fn(batch)
    data['deltaE'] = torch.stack([b['deltaE'] for b in batch])
    data['sigma'] = torch.stack([b['sigma'] for b in batch])
    data['fermi'] = torch.stack([b['fermi'] for b in batch])
    data['temp'] = torch.stack([b['temp'] for b in batch])
    data['synth_diff'] = torch.stack([b['synth_diff'] for b in batch])
    return data

class LabeledDataModule:
    def __init__(self, path: str, batch_size: int = 64, num_workers: int = 4):
        self.path = Path(path)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        ds = LabeledDataset(self.path)
        n = len(ds)
        # 确保训练集和验证集都有数据
        train_size = max(1, int(0.9 * n))
        val_size = max(1, n - train_size)
        self.train_ds, self.val_ds = random_split(ds, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn_label)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_label) 