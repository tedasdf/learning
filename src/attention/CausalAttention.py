"""
Causal Self-Attention module with configurable output dimension.

link:https://arxiv.org/pdf/1706.03762
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


@dataclass
class AttnConfig:
    d_model: int
    n_head: int
    block_size: int
    dropout: float



class CausalSelfAttention(nn.Module): 
    def __init__(self, cfg: AttnConfig , output_dim = None):
        super().__init__()
        
        self.d_model = cfg.d_model
        # Override head_dim and projection layers
        self.head_dim = self.d_model // cfg.n_head # mutli-head attention
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(self.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(self.d_model, output_dim)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop= nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

        

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()

        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        
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
    attn = CausalSelfAttention(attn_cfg, output_dim=16)
    out = attn(x)
    print(out.shape)  # (2, 5, 16)
