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
    intermediate_dim: int  # must be specified for bottleneck attention



class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AttnConfig , output_dim = None):
        super().__init__()
        
        if cfg.intermediate_dim == 0:
            self.intermediate_dim  = cfg.d_model
        else :
            self.intermediate_dim = cfg.intermediate_dim 
        assert self.intermediate_dim % cfg.n_head == 0, "bottleneck_dim must be divisible by n_head"

        # Override head_dim and projection layers
        self.head_dim = self.intermediate_dim // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * self.intermediate_dim)
        self.proj = nn.Linear(self.intermediate_dim, output_dim)
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
        y = y.transpose(1, 2).contiguous().view(B, T, self.intermediate_dim)
        return self.resid_drop(self.proj(y))
    
if __name__ == "__main__":
    # Example usage
    x = torch.randn(2, 5, 16)  # batch=2, seq_len=5, embed_dim=16
    attn = CausalSelfAttention(embed_dim=16, num_heads=4)
    out = attn(x)
    print(out.shape)  # (2, 5, 16)
