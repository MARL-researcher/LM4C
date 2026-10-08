import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch as th

class SoftAttention(nn.Module):
    def __init__(self, args, num_heads=2):
        super(SoftAttention, self).__init__()

        self.trajectory_dim = args.rnn_hidden_dim
        self.state_dim = int(np.prod(args.state_shape))

        self.out_dim = args.rnn_hidden_dim

        assert self.out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = self.out_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.q_proj = nn.Linear(self.state_dim, self.out_dim)
        self.k_proj = nn.Linear(self.trajectory_dim, self.out_dim)
        self.v_proj = nn.Linear(self.trajectory_dim, self.out_dim)

        self.out_proj = nn.Linear(self.out_dim, self.out_dim)

        device = "cuda" if args.use_cuda else "cpu"
        if args.mask_state:
            mask = 1 - th.eye(args.n_agents)
            self.mask = (mask.unsqueeze(0).repeat(args.batch_size, 1, 1)).unsqueeze(1).unsqueeze(2).to(device=device)

    def forward(self, x, g, mask=False):
        """
        x: [B, T, N, DIM]  -> agent trajectories
        g: [B, T, DIM]     -> global state
        return: [B, T, DIM] -> aggregated global representation
        """
        B, T, N, D = x.shape
        H = self.num_heads
        d_h = self.head_dim
        d_out = self.out_dim

        q = self.q_proj(g)          # [B, T, D]
        k = self.k_proj(x)          # [B, T, N, D]
        v = self.v_proj(x)          # [B, T, N, D]

        q = q.view(B, T, H, d_h).permute(0, 2, 1, 3)       # [B, H, T, d_h]
        k = k.view(B, T, N, H, d_h).permute(0, 3, 1, 2, 4) # [B, H, T, N, d_h]
        v = v.view(B, T, N, H, d_h).permute(0, 3, 1, 2, 4) # [B, H, T, N, d_h]

        q = q.unsqueeze(3)                                  # [B, H, T, 1, d_h]
        attn = th.matmul(q, k.transpose(-1, -2)) / self.scale  # [B, H, T, 1, N]

        attn = F.softmax(attn, dim=-1)                      # [B, H, T, 1, N]

        if mask:
            attn = attn * self.mask
            s = th.matmul(attn, v)                # [B, H, T, N, d_h]
            s = s.permute(0, 2, 3, 1, 4).contiguous().view(B, T, N, d_out)  # [B, T, N, D]
            s = self.out_proj(s)                                # [B, T, N, D]
            return s
        else:
            s = th.matmul(attn, v).squeeze(3)                # [B, H, T, d_h]
            s = s.permute(0, 2, 1, 3).contiguous().view(B, T, d_out)  # [B, T, D]
            s = self.out_proj(s)                                # [B, T, D]
            return s