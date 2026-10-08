import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from components.consensus import LocalConsensusGenerator


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention for inputs [B, N, D]
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, alive_mask):
        B, N, D = x.size()
        H = self.nhead

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to [B, H, N, head_dim]
        def reshape(t):
            return t.view(B, N, H, self.head_dim).transpose(1, 2)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        # attention scores: [B, H, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = alive_mask.transpose(1, 2).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # [B, H, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out

class TransformerEncoderLayer(nn.Module):
    """
    Pre-LN Transformer Encoder Layer for [B, N, D].
    """
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

    def forward(self, x, alive_mask):
        # --- Self Attention ---
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, alive_mask)
        x = x + attn_out

        # --- FFN ---
        x_norm = self.norm2(x)
        ff_out = self.linear2(self.activation(self.linear1(x_norm)))
        x = x + ff_out

        return x

class TransformerEncoder(nn.Module):
    """
    multi-layer transformer encoder, num_layers = communication rounds
    """
    def __init__(self, input_dim, dim_feedforward, args):
        super().__init__()

        d_model = args.rnn_hidden_dim
        nhead = args.att_heads
        num_layers = args.comm_rounds
        
        self.encoder = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward
            )
            for _ in range(num_layers)
        ])

        self.local_consensus = LocalConsensusGenerator(args=args)

    def forward(self, x, alive_mask):
        
        comm_info = self.encoder(x)

        for layer in self.layers:
            comm_info = layer(comm_info, alive_mask)
        
        latent_z = self.local_consensus(comm_info)

        return comm_info, latent_z
