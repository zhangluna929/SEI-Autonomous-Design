import torch
import torch.nn as nn
import torch.nn.functional as F
from ..pretrain.models import MultimodalModel

class SpectralAttentionGate(nn.Module):
    """简单门控：fused + g * spectral (g 由两者拼接后线性 -> sigmoid)"""
    def __init__(self, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim * 2, 1)

    def forward(self, fused, spec):
        gate = torch.sigmoid(self.fc(torch.cat([fused, spec], dim=-1)))
        return fused + gate * spec

class Predictor(nn.Module):
    def __init__(self, vocab_size, freeze_ratio=0.7):
        super().__init__()
        self.encoder = MultimodalModel(vocab_size)
        # Freeze first 70 % parameters (roughly)
        params = list(self.encoder.parameters())
        freeze_until = int(len(params) * freeze_ratio)
        for p in params[:freeze_until]:
            p.requires_grad = False

        self.gate = SpectralAttentionGate()
        self.delta_head = nn.Linear(256, 1)
        self.sigma_head = nn.Linear(256, 1)
        self.fermi_head = nn.Linear(256, 1)
        self.temp_head = nn.Linear(256, 1)
        self.diff_head = nn.Linear(256, 1)
        # Task uncertainty (log sigma^2) learnable params
        self.log_vars = nn.Parameter(torch.zeros(5))

    def forward(self, batch):
        out = self.encoder(batch)
        fused, (_, _, spec), node_logits, patch_tokens = out
        gated = self.gate(fused, spec)
        delta_pred = self.delta_head(gated).squeeze(-1)
        sigma_pred = self.sigma_head(gated).squeeze(-1)
        fermi_pred = self.fermi_head(gated).squeeze(-1)
        temp_pred = self.temp_head(gated).squeeze(-1)
        diff_pred = self.diff_head(gated).squeeze(-1)
        return delta_pred, sigma_pred, fermi_pred, temp_pred, diff_pred

    def loss(self, preds, targets):
        delta_pred, sigma_pred, fermi_pred, temp_pred, diff_pred = preds
        delta_t, sigma_t, fermi_t, temp_t, diff_t = targets
        # Huber for deltaE
        l_delta = F.huber_loss(delta_pred, delta_t, delta=0.1)
        # log regression for sigma (we assume sigma_t 已取 log)
        l_sigma = F.mse_loss(sigma_pred, sigma_t)
        l_fermi = F.binary_cross_entropy_with_logits(fermi_pred, fermi_t)
        l_temp = F.mse_loss(temp_pred, temp_t)
        l_diff = F.mse_loss(diff_pred, diff_t)
        losses = torch.stack([l_delta, l_sigma, l_fermi, l_temp, l_diff])
        precision = torch.exp(-self.log_vars)
        weighted = precision * losses + self.log_vars
        return weighted.sum(), {'delta': l_delta, 'sigma': l_sigma, 'fermi': l_fermi, 'temp': l_temp, 'diff': l_diff} 