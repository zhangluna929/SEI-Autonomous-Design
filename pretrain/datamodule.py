import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from pathlib import Path
import random
import numpy as np
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader as GeoLoader
import selfies as sf
from torch.nn import functional as F

MASK_ID = 0
CLS_ID = 1
PAD_ID = 2
MAX_SEQ_LEN = 256
PATCH_SIZE = 40  # aligned with SpectraEncoder

def selfies_tokenize(s: str) -> List[int]:
    tokens = sf.split_selfies(s)
    ids = [CLS_ID]
    for t in tokens[: MAX_SEQ_LEN - 1]:
        ids.append(abs(hash(t)) % 30000 + 10)  # simple hash vocab
    if len(ids) < MAX_SEQ_LEN:
        ids += [PAD_ID] * (MAX_SEQ_LEN - len(ids))
    return ids

class MultimodalDataset(Dataset):
    def __init__(self, parquet_path: Path):
        table = pq.read_table(parquet_path)
        self.df = table.to_pandas()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Crystal graph
        graph_list: List[Dict[str, Any]] = row.get('crystal_graph') or []
        if graph_list:
            num_nodes = int(max(max(g['site'], g['neighbor']) for g in graph_list) + 1)
        else:
            num_nodes = 1
        x = torch.ones((num_nodes, 1), dtype=torch.float)
        edge_index = torch.tensor([[g['site'] for g in graph_list], [g['neighbor'] for g in graph_list]], dtype=torch.long) if graph_list else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor([[g['weight']] for g in graph_list], dtype=torch.float) if graph_list else torch.empty((0, 1), dtype=torch.float)
        crystal_data = GeoData(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # --- Node masking ---
        node_mask = torch.rand(num_nodes) < 0.15
        x[node_mask] = 0.0  # masked nodes feature zeroed
        node_target = node_mask.float()  # 1 means masked

        # Polymer sequence
        selfies_str = row.get('selfies') or ''
        ids = selfies_tokenize(selfies_str)
        token_ids = torch.tensor(ids, dtype=torch.long)
        # Create mask
        attn_mask = token_ids == PAD_ID
        mask_positions = (torch.rand_like(token_ids.float()) < 0.15) & (~attn_mask)
        target_tokens = token_ids.clone()
        token_ids[mask_positions] = MASK_ID

        # --- Spectra patch masking ---
        spec = np.array(row.get('spectrum') or [0.0], dtype=np.float32)
        spectra_orig = torch.tensor(spec, dtype=torch.float)
        spectra_tensor = spectra_orig.clone()
        num_patches = spectra_tensor.shape[0] // PATCH_SIZE
        if num_patches > 0:
            patch_mask = torch.rand(num_patches) < 0.15
            for i, m in enumerate(patch_mask):
                if m:
                    start = i * PATCH_SIZE
                    spectra_tensor[start : start + PATCH_SIZE] = 0.0
        else:
            patch_mask = torch.zeros(0, dtype=torch.bool)

        return {
            'crystal': crystal_data,
            'seq_ids': token_ids,
            'seq_target': target_tokens,
            'attn_mask': attn_mask,
            'spectra': spectra_tensor,
            'spectra_orig': spectra_orig,
            'node_mask': node_mask,
            'patch_mask': patch_mask,
        }


def collate_fn(batch):
    # Crystal graphs use pyg loader later; here we separate
    crystals = [b['crystal'] for b in batch]
    from torch_geometric.loader import Batch as GeoBatch
    crystal_batch = GeoBatch.from_data_list(crystals)
    seq_ids = torch.stack([b['seq_ids'] for b in batch])
    seq_target = torch.stack([b['seq_target'] for b in batch])
    attn_mask = torch.stack([b['attn_mask'] for b in batch])
    # pad spectra to same length
    max_len = max([b['spectra'].shape[0] for b in batch])
    spectra_batch = torch.stack([F.pad(b['spectra'], (0, max_len - b['spectra'].shape[0])) for b in batch])
    # pad patch mask similarly
    max_patch = max([b['patch_mask'].shape[0] for b in batch])
    patch_masks = torch.stack([F.pad(b['patch_mask'].float(), (0, max_patch - b['patch_mask'].shape[0])) for b in batch]) > 0
    node_masks = torch.cat([b['node_mask'] for b in batch])
    return {
        'x_c': crystal_batch.x,
        'edge_index': crystal_batch.edge_index,
        'edge_attr': crystal_batch.edge_attr,
        'batch': crystal_batch.batch,
        'seq_ids': seq_ids,
        'seq_target': seq_target,
        'attn_mask': attn_mask,
        'spectra': spectra_batch,
        'spectra_orig': torch.stack([F.pad(b['spectra_orig'], (0, max_len - b['spectra_orig'].shape[0])) for b in batch]),
        'node_mask': node_masks,
        'patch_mask': patch_masks,
    }

class ParquetDataModule:
    def __init__(self, path: str, batch_size: int = 64, num_workers: int = 4):
        self.path = Path(path)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        ds = MultimodalDataset(self.path)
        n = len(ds)
        self.train_ds, self.val_ds = torch.utils.data.random_split(ds, [int(0.95 * n), n - int(0.95 * n)])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn) 