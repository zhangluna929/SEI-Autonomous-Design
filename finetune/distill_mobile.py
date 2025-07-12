import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from ..pretrain.datamodule import ParquetDataModule
from ..pretrain.models import MultimodalModel, PATCH_SIZE
from .mobile_encoder import MobileSpectraEncoder

class DistillModule(pl.LightningModule):
    def __init__(self, teacher_ckpt, vocab_size=30010):
        super().__init__()
        self.teacher = MultimodalModel(vocab_size)
        self.teacher.load_state_dict(torch.load(teacher_ckpt, map_location='cpu'))
        for p in self.teacher.parameters():
            p.requires_grad = False
        # replace spectra encoder with mobile student
        self.student = MobileSpectraEncoder()
        self.proj = torch.nn.Linear(256, 256)  # match dim
        self.temperature = 2.0

    def forward(self, batch):
        with torch.no_grad():
            teacher_emb, *_ = self.teacher(batch)
        student_emb = self.proj(self.student(batch['spectra']))
        return student_emb, teacher_emb

    def training_step(self, batch, batch_idx):
        student_emb, teacher_emb = self(batch)
        loss = F.mse_loss(student_emb, teacher_emb.detach())
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        return [opt], [sch]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='master.parquet')
    parser.add_argument('--teacher', default='encoder.ckpt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()

    dm = ParquetDataModule(args.data, batch_size=args.batch)
    dm.setup()

    model = DistillModule(args.teacher)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='gpu', devices=1, precision='16-mixed', default_root_dir='distill_logs')
    trainer.fit(model, dm)
    torch.save(model.student.state_dict(), 'encoder_mobile.ckpt') 