import torch
import torch.nn as nn
from einops import rearrange

from ..utils import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register()
class VanillaSelfAttention(nn.Module):
    """
    Vanilla O(n^2) Self attention

    Parameters
    ----------
    dim: int
        Dimension of the embedding
    num_heads: int
        Number of the attention heads
    head_dim: int
        Dimension of each head
    p_dropout: float
        Dropout Probability

    """

    def __init__(self, dim, num_heads=8, head_dim=64, p_dropout=0.0):
        super().__init__()

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == dim)

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying self-attention on input tensor

        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)
