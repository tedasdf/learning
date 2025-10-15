
import torch
import torch.nn as nn

from dataclasses import dataclass

from algorithm.attention.GroupedQueryAttention import GroupedQueryAttention
from algorithm.attention.CausalAttention import AttnConfig

@dataclass
class QwenConfig:
    hidden_size : 896
    layers: 24 
    query_heads: 14
    kv_heads: 2
    head_size: 64
    intermediate_size :  4864
    embedding_tying: True
    vocab_size: 151646

@dataclass
class GroupAttnConfig:
    query_heads: 14


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features if hidden_features is not None else in_features * 2
        self.out_features = out_features if out_features is not None else in_features

        self.w1 = nn.Linear(self.in_features, self.hidden_features)
        self.w2 = nn.Linear(self.in_features, self.hidden_features)
        self.out = nn.Linear(self.hidden_features, self.out_features)
        self.silu = nn.SiLU()

    def forward(self, x):
        gate = self.w1(x)
        linear_part = self.w2(x)
        activated_gate = self.silu(gate)
        return self.out(activated_gate * linear_part)
    
class QwenModel(nn.Module):
    def __init__(self, config: QwenConfig ):

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.drop      = nn.Dropout(cfg.dropout)

        self.block_list = nn.ModuleList()
        in_size = config.vocab_size 
        
        for lay in config.layers:
            self.block_list.append(
                Block()
            )

        self.ln_f      = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(config.hidden_size, config.vocab_size , bias= False)
        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        hidden_features = int(cfg.d_model * 8 / 3)

        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_features),
            SwiGLU(hidden_features),          # hidden_features → out_features defaults to hidden_features
            nn.Linear(hidden_features, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )


    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


if __name__ == '__main__':

    # Example usage
    x = torch.randn(2, 5, 16)  # batch=2, seq_len=5, embed_dim=16
    print(x.shape)
    attn_cfg = AttnConfig(d_model=16, n_head=4, block_size=10, dropout=0.1)
    attn = GroupedQueryAttention(attn_cfg, group_size=2, output_dim=16)
    out = attn(x)
    print(out.shape)  # (2, 5, 16)
