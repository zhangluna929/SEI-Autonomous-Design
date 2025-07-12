import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.depth = nn.Conv1d(in_ch, in_ch, k, stride, padding=k//2, groups=in_ch, bias=False)
        self.point = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return F.relu6(self.bn(x))

class MobileSpectraEncoder(nn.Module):
    """Lightweight 1-D CNN encoder returning 256-dim global embedding"""
    def __init__(self, in_len=4000, base_ch=32, emb_dim=256):
        super().__init__()
        layers = [nn.Conv1d(1, base_ch, 3, padding=1, bias=False), nn.BatchNorm1d(base_ch), nn.ReLU6()]
        ch = base_ch
        for _ in range(4):
            layers.append(DepthwiseSeparableConv(ch, ch*2, k=3, stride=2))
            ch *= 2
        self.cnn = nn.Sequential(*layers)
        # compute feature length
        dummy = torch.zeros(1,1,in_len)
        with torch.no_grad():
            feat_len = self.cnn(dummy).shape[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch, emb_dim)
    def forward(self, x):  # B,L
        x = x.unsqueeze(1)
        feat = self.cnn(x)
        pooled = self.pool(feat).squeeze(-1)
        return self.proj(pooled) 