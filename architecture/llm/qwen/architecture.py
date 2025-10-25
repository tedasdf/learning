
import torch
import torch.nn as nn

from dataclasses import dataclass

from algorithm.attention.GroupedQueryAttention import GroupedQueryAttention
from algorithm.attention.CausalAttention import AttnConfig
from algorithm.layer.LayerNorm import RMSNorm

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
    kv_heads: 2
    head_size: 64


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
    

class Qwen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Qwen2Config] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor
    
    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
class QwenModel(nn.Module):
    def __init__(self, config: QwenConfig ):

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Parameter(torch.zeros(1, config.block_size, config.d_model))
        self.drop      = nn.Dropout(config.dropout)

        self.block_list = nn.ModuleList()
        in_size = config.vocab_size 
        
        for lay in config.layers:
            self.block_list.append(
                Block()
            )
        
        self.norm = RMSNorm()
        self.rotary_emb = RotaryEmbedding()

        self.ln_f      = nn.LayerNorm(config.d_model)
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
