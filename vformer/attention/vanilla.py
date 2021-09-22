import torch
import torch.nn as nn
from einops import rearrange


class VanillaSelfAttention(nn.Module):
    """
    Vanilla Self attention
    Parameters:
    -----------
    dim: int
        Dimension of the Embedding
    heads: int
        Number of attention head
    dim_head: int
        Dimension of the head
    p_dropout: float
        Probability for dropout layer
    """

    def __init__(self, dim, heads=8, dim_head=64, p_dropout=0.0):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.tensor):
        # x.shape= B,N,C

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        # out.shape=B,N,C

        return self.to_out(out)
