import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_stable_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    mask = mask.bool()
    neg_inf = -1e9
    scores_masked = scores.masked_fill(~mask, neg_inf)
    max_score = scores_masked.max(dim=dim, keepdim=True).values
    scores_centered = scores_masked - max_score
    exp_scores = torch.exp(scores_centered) * mask.to(dtype=scores.dtype)
    exp_sum = exp_scores.sum(dim=dim, keepdim=True)
    probs = exp_scores / (exp_sum + eps)
    return probs


class NeighborAttention(nn.Module):
    def __init__(self, emb_dim: int, head_dim: int, num_heads: int = 1):
        super().__init__()
        assert emb_dim > 0
        assert num_heads >= 1

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (emb_dim // num_heads)
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads unless you pass head_dim explicitly"

        self.to_q = nn.Linear(emb_dim, num_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(emb_dim, num_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(emb_dim, num_heads * self.head_dim, bias=False)

        self.to_out = nn.Linear(num_heads * self.head_dim, emb_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, E = x.shape
        mask = adj.bool()

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        def reshape_for_heads(tensor):
            return tensor.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_for_heads(q)
        k = reshape_for_heads(k)
        v = reshape_for_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask_expanded = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn = masked_stable_softmax(scores, mask_expanded, dim=-1)

        out_per_head = torch.matmul(attn, v)
        out = out_per_head.transpose(1, 2).reshape(B, N, -1)
        out = self.to_out(out)
        return out