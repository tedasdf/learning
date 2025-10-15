"""
Grouped Query Attention module.

link:https://arxiv.org/pdf/2305.13245
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .CausalAttention import CausalSelfAttention, AttnConfig



class GroupedQueryAttention(CausalSelfAttention):
    def __init__(self, cfg: AttnConfig, group_size: int  , output_dim = None):
        super().__init__(cfg, output_dim)
        self.group_size = group_size
        self.repeat_factor = self.n_head // self.group_size

        self.q = nn.Linear(self.d_model, cfg.d_model)
        self.kv = nn.Linear(self.d_model, 2 * self.group_size * self.head_dim)


    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        
        q = self.q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        kv = self.kv(x).view(B, T, 2, self.group_size, self.head_dim).transpose(1, 3)

        k, v = kv[..., 0, :, :], kv[..., 1, :, :]

       
        k = k.repeat_interleave(self.repeat_factor, dim=1)
        v = v.repeat_interleave(self.repeat_factor, dim=1)
    
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
    attn = GroupedQueryAttention(attn_cfg, group_size=2, output_dim=16)
    out = attn(x)
    print(out.shape)  # (2, 5, 16)
