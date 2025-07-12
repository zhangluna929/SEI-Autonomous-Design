import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.nn.norm import BatchNorm

PATCH_SIZE = 40

# ----------------- Crystal Encoder (modified CGCNN) -------------------
class CrystalEncoder(nn.Module):
    def __init__(self, node_feat=64, edge_feat=64, num_layers=6, out_dim=256):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(CGConv(node_feat, dim=edge_feat))
            self.norms.append(BatchNorm(node_feat))
        self.readout = nn.Linear(node_feat, out_dim)
        self.node_head = nn.Linear(node_feat, 1)  # predict mask

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.norms):
            x = F.relu(bn(conv(x, edge_index, edge_attr)))
        node_logits = self.node_head(x).squeeze(-1)  # per node
        pooled = global_mean_pool(x, batch)
        return self.readout(pooled), node_logits

# ----------------- Polymer Encoder (Transformer) ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, out_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, token_ids, attn_mask=None):
        x = self.embed(token_ids)
        x = self.pos(x)
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        # 取 CLS (假设第一位是 CLS)
        cls = x[:, 0, :]
        return self.proj(cls)

# ----------------- 1D ViT for spectra --------------------------------
class PatchEmbed1D(nn.Module):
    def __init__(self, seq_len=4000, patch_size=40, in_chans=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        num_patches = seq_len // patch_size
        self.proj = nn.Linear(patch_size * in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):  # B, 1, L
        B, C, L = x.shape
        x = x.view(B, C, L // self.patch_size, self.patch_size)  # B,C,n,patch
        x = x.flatten(2)  # B,C,n*patch
        x = x.transpose(1, 2)  # B, n , C*patch
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        return x

class SpectraEncoder(nn.Module):
    def __init__(self, seq_len=4000, patch_size=PATCH_SIZE, embed_dim=256, depth=6, nhead=8, out_dim=256):
        super().__init__()
        self.patch = PatchEmbed1D(seq_len, patch_size, 1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x):  # B,L
        x = x.unsqueeze(1)  # B,1,L
        tok = self.patch(x)  # B, n_p+1, dim
        tok = self.transformer(tok)
        cls = tok[:, 0, :]
        patch_tokens = tok[:, 1:, :]
        return self.proj(cls), patch_tokens  # patch_tokens shape B, n_p, dim

# ----------------- Cross-Attention Fusion -----------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, cris, poly, spec):
        # 将三个向量扩展为序列长度 1
        seq = torch.stack([cris, poly, spec], dim=1)  # B,3,dim
        fused, _ = self.attn(seq, seq, seq)
        fused = fused.mean(dim=1)  # B,dim
        return fused

# ----------------- Overall Model --------------------------------------
class MultimodalModel(nn.Module):
    def __init__(self, vocab_size, **kwargs):
        super().__init__()
        self.crystal = CrystalEncoder(**kwargs.get('crystal', {}))
        self.polymer = SequenceEncoder(vocab_size, **kwargs.get('polymer', {}))
        self.spectra = SpectraEncoder(**kwargs.get('spectra', {}))
        self.fusion = CrossAttentionFusion()
        self.proj = nn.Linear(256, 256)  # projection for contrastive space
        self.patch_recon = nn.Linear(256, PATCH_SIZE)  # use PATCH_SIZE constant

    def forward(self, batch):
        # crystal returns tuple
        z_c, node_logits = self.crystal(batch['x_c'], batch['edge_index'], batch['edge_attr'], batch['batch'])
        z_p = self.polymer(batch['seq_ids'], batch['attn_mask'])
        z_s, patch_tokens = self.spectra(batch['spectra'])
        fused = self.fusion(z_c, z_p, z_s)
        return self.proj(fused), (z_c, z_p, z_s), node_logits, patch_tokens 