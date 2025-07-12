import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .predictor import Predictor
from .datamodule import LabeledDataModule

class PredictorModule(pl.LightningModule):
    def __init__(self, vocab_size=30010):
        super().__init__()
        self.model = Predictor(vocab_size)

    def training_step(self, batch, batch_idx):
        preds = self.model(batch)
        targets = (batch['deltaE'], batch['sigma'], batch['fermi'], batch['temp'], batch['synth_diff'])
        loss, losses_dict = self.model.loss(preds, targets)
        self.log_dict({'train_' + k: v for k, v in losses_dict.items()})
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model(batch)
        targets = (batch['deltaE'], batch['sigma'], batch['fermi'], batch['temp'], batch['synth_diff'])
        loss, losses_dict = self.model.loss(preds, targets)
        self.log_dict({'val_' + k: v for k, v in losses_dict.items()})
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='labeled.parquet')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    dm = LabeledDataModule(args.data, batch_size=args.batch)
    dm.setup()

    pl.seed_everything(42)
    model = PredictorModule()

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator='gpu', devices=1, precision='16-mixed', default_root_dir='finetune_logs', log_every_n_steps=20)
    trainer.fit(model, dm)
    trainer.save_checkpoint('predictor.ckpt') 