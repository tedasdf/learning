"""
Grouped Query Attention module.

link:https://arxiv.org/pdf/2305.13245
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from .CausalAttention import CausalSelfAttention, AttnConfig

@dataclass
class GroupAttnConfig(AttnConfig):
    query_heads: int
    kv_heads: int
    head_size: int


class GroupedQueryAttention(CausalSelfAttention):
    def __init__(self, cfg: GroupAttnConfig, output_dim=None):
        super().__init__(cfg, output_dim)
        self.query_heads = cfg.query_heads
        self.kv_heads = cfg.kv_heads
        self.group_size = cfg.query_heads // cfg.kv_heads
        self.head_size = cfg.head_size

        self.q = nn.Linear(self.d_model, self.query_heads * self.head_size)
        self.kv = nn.Linear(self.d_model, 2 * self.kv_heads * self.head_size)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.q(x).view(B, T, self.query_heads, self.head_size).transpose(1, 2)
        kv = self.kv(x).view(B, T, self.kv_heads, 2, self.head_size)
        k, v = kv[..., 0, :], kv[..., 1, :]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        idx = torch.arange(self.query_heads, device=x.device) // self.group_size
        K_sel = k[:, idx, :, :]
        V_sel = v[:, idx, :, :]

        att = (q @ K_sel.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ V_sel
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.resid_drop(self.proj(y))

    
if __name__ == "__main__":
    # Example input
    x = torch.randn(2, 5, 128)  # (batch=2, seq_len=5, embed_dim=128)

    # Define grouped attention configuration
    attn_cfg = GroupAttnConfig(
        query_heads=8,   # total query heads
        kv_heads=2,      # total key/value heads
        head_size=16     # per-head dimension
        d_model=256,
        n_head=None,
        block_size=

    )

    # Create attention module
    attn = GroupedQueryAttention(attn_cfg, output_dim=128)

    # Forward pass
    out = attn(x)
    print(out.shape)  # (2, 5, 128)
