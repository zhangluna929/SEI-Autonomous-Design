import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from .models import MultimodalModel

MASK_ID = 0
PATCH_SIZE = 40

class PretrainModule(pl.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = MultimodalModel(vocab_size)
        self.temperature = 0.1

    def forward(self, batch):
        z, _ = self.model(batch)
        return z

    def _contrastive_loss(self, z):
        # z: B,256
        z = F.normalize(z, dim=1)
        B = z.size(0)
        # positive pairs: (i,i) vs negative others (shuffled)
        idx = torch.randperm(B, device=self.device)
        z2 = z[idx]
        logits = torch.mm(z, z2.t()) / self.temperature
        labels = torch.arange(B, device=self.device)
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        fused, (_, _, _), node_logits, patch_tokens = out
        
        # Reconstruction loss (polymer tokens) - 修复实现
        masked_positions = batch['seq_ids'] == MASK_ID
        if masked_positions.any():
            # 使用模型的输出来计算重建损失
            vocab_size = self.model.polymer.embed.num_embeddings
            # 创建简单的分类头来预测被mask的token
            if not hasattr(self, 'token_classifier'):
                self.token_classifier = nn.Linear(256, vocab_size).to(self.device)
            
            # 获取被mask位置的特征
            polymer_features = self.model.polymer(batch['seq_ids'], batch['attn_mask'])
            logits = self.token_classifier(polymer_features.unsqueeze(1).expand(-1, batch['seq_ids'].size(1), -1))
            
            # 计算重建损失
            target_tokens = batch['seq_target'][masked_positions]
            pred_logits = logits[masked_positions]
            rec_loss = F.cross_entropy(pred_logits, target_tokens)
        else:
            rec_loss = torch.tensor(0.0, device=self.device)
        
        # 对比学习损失
        contrast_loss = self._contrastive_loss(fused)
        
        # 节点重建损失
        node_mask = batch['node_mask']
        if node_mask.numel() > 0:
            l_node = F.binary_cross_entropy_with_logits(node_logits, node_mask)
        else:
            l_node = torch.tensor(0.0, device=self.device)
        
        # 光谱patch重建损失
        patch_mask = batch['patch_mask']  # B,n_p
        if patch_mask.any() and patch_tokens.size(1) > 0:
            try:
                # 确保维度匹配
                batch_size = batch['spectra_orig'].size(0)
                spec_len = batch['spectra_orig'].size(1)
                n_patches = spec_len // PATCH_SIZE
                
                if n_patches > 0 and patch_tokens.size(1) >= n_patches:
                    target_spec = batch['spectra_orig'][:, :n_patches*PATCH_SIZE].view(batch_size, n_patches, PATCH_SIZE)
                    pred_patch = self.model.patch_recon(patch_tokens[:, :n_patches, :])  # B,n_p,patch_size
                    
                    # 只计算被mask的patch的损失
                    valid_mask = patch_mask[:, :n_patches]
                    if valid_mask.any():
                        mse = (pred_patch - target_spec) ** 2
                        l_patch = (mse * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum().clamp_min(1.0)
                    else:
                        l_patch = torch.tensor(0.0, device=self.device)
                else:
                    l_patch = torch.tensor(0.0, device=self.device)
            except Exception as e:
                # 如果计算失败，使用零损失
                l_patch = torch.tensor(0.0, device=self.device)
        else:
            l_patch = torch.tensor(0.0, device=self.device)
        
        # 总损失
        loss = rec_loss + contrast_loss + l_node + l_patch
        
        # 记录损失
        self.log_dict({
            'loss': loss, 
            'rec_loss': rec_loss, 
            'contrast_loss': contrast_loss,
            'node_loss': l_node,
            'patch_loss': l_patch
        })
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4, weight_decay=1e-2)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2),
            'interval': 'epoch',
        }
        return [optimizer], [scheduler]

if __name__ == '__main__':
    import argparse
    from .datamodule import ParquetDataModule

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='master.parquet')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    dm = ParquetDataModule(args.data, batch_size=args.batch)
    dm.setup()
    vocab_size = 30010  # hash vocab size
    model = PretrainModule(vocab_size)

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='gpu', devices=1, strategy='ddp_spawn', accumulate_grad_batches=2, precision='16-mixed', default_root_dir='pretrain_logs', gradient_clip_val=1.0, log_every_n_steps=50)
    trainer.fit(model, dm)
    trainer.save_checkpoint('encoder.ckpt') 