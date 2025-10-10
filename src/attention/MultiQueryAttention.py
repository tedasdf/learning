"""
Multi-Query Attention module.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention.CausalAttention import CausalSelfAttention, AttnConfig
from typing import Optional


class MultiQueryAttention(CausalSelfAttention):
    def __init__(self, cfg: AttnConfig , output_dim = None):
        super().__init__(cfg, output_dim)
        # Override head_dim and projection layers
        self.head_dim = cfg.d_model // cfg.n_head # mutli-head attention
        self.n_head   = cfg.n_head
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.kv = nn.Linear(cfg.d_model, 2 * self.head_dim)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        kv = self.kv(x).view(B, T, 2, 1, self.head_dim).transpose(1, 3)

        k, v = kv[..., 0, :, :], kv[..., 1, :, :]
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.resid_drop(self.proj(y))

if __name__ == "__main__":
    # Example usage
    x = torch.randn(2, 5, 16)  # batch=2, seq_len=5, embed_dim=16
    print(x.shape)
    attn_cfg = AttnConfig(d_model=16, n_head=4, block_size=10, dropout=0.1)
    attn = MultiQueryAttention(attn_cfg, output_dim=16)
    out = attn(x)
    print(out.shape)  # (2, 5, 16)