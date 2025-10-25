import torch
import torch.nn as nn
from typing import Optional



class RoPE:
    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin):
        # q, k: [batch_size, seq_len, num_heads, head_dim]
        # cos, sin: [seq_len, head_dim]
        # Apply rotary position embedding to query and key tensors.
        q_cos = q * cos[:, None, :, :]
        q_sin = q * sin[:, None, :, :]
        k_cos = k * cos[:, None, :, :]
        k_sin = k * sin[:, None, :, :]

        q_rotated = q_cos + RoPE.rotate_half(q_sin)
        k_rotated = k_cos + RoPE.rotate_half(k_sin)

        return q_rotated, k_rotated

    @staticmethod
    def rotate_half(x):
        # x: [batch_size, seq_len, num_heads, head_dim]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    




def simplified_case( d , seq_len , theta = 0.1):
    # Settings
    '''
        d : dimension
        seq_len: number of positions 
        theta: preset rotation constant
    '''
    # RoPE rotation
    positions = torch.arange(seq_len).float()  # [0, 1, ..., seq_len-1]
    cos_mtheta = torch.cos(positions * theta)  # shape [seq_len]
    sin_mtheta = torch.sin(positions * theta)

    # Function to rotate a 2D vector
    def rotate_2d(x, cos_val, sin_val):
        x1, x2 = x[..., 0], x[..., 1]
        x1_rot = cos_val * x1 - sin_val * x2
        x2_rot = sin_val * x1 + cos_val * x2
        return torch.stack([x1_rot, x2_rot], dim=-1)

    # Apply rotation to Q and K
    Q_rot = torch.stack([rotate_2d(Q[i], cos_mtheta[i], sin_mtheta[i]) for i in range(seq_len)])
    K_rot = torch.stack([rotate_2d(K[i], cos_mtheta[i], sin_mtheta[i]) for i in range(seq_len)])

    return Q_rot , K_rot
    # Compute g(x_m, x_n, m-n) using complex representation (real part only)
    # # Treat 2D vector as complex: (x1 + i*x2)
    # def g(Qr, Kr):
    #     seq_len = Qr.shape[0]
    #     G = torch.zeros(seq_len, seq_len)
    #     for m in range(seq_len):
    #         for n in range(seq_len):
    #             q_complex = Qr[m, 0] + 1j * Qr[m, 1]
    #             k_complex = Kr[n, 0] + 1j * Kr[n, 1]
    #             G[m, n] = torch.real(q_complex * torch.conj(k_complex) * torch.exp(1j * (m-n) * theta))
    #     return G

    # G = g(Q_rot, K_rot)
    # print("g matrix:\n", G)



def more_general(Q, K, theta=0.1):
    d = Q.shape[1]
    seq_len = Q.shape[0]

    def rotate_vec(x, angle):
        R = torch.zeros(d, d)
        for i in range(0, d, 2):
            c, s = torch.cos(angle), torch.sin(angle)
            R[i:i+2, i:i+2] = torch.tensor([[c, -s], [s, c]])
        return R @ x

    Q_rot = torch.stack([rotate_vec(Q[i], i*theta) for i in range(seq_len)])
    K_rot = torch.stack([rotate_vec(K[i], i*theta) for i in range(seq_len)])
    return Q_rot, K_rot


if __name__ == "__main__":
    d = 2 
    seq_len = 5
    
    # Sample input embeddings (for all positions)
    X = torch.rand(seq_len, d)  # shape [seq_len, d], (x_m^1, x_m^2)

    # Learnable linear weights for q and k
    Wq = nn.Linear(d, d, bias=False)
    Wk = nn.Linear(d, d, bias=False)

    Q , K = Wq(X), Wk(X)
    print("======Q Shape , K Shape ==========")
    print(Q.shape , K.shape)

    if d == 2:
        Q_rot, K_rot = simplified_case(Q, K, seq_len, theta=0.1)
    else:
        Q_rot, K_rot = more_general(Q, K, theta=0.2)
        
    print("Q_rot:\n", Q_rot)
    print("K_rot:\n", K_rot)
    print("Original Q norms:", Q.norm(dim=1))
    print("Rotated Q norms:", Q_rot.norm(dim=1))