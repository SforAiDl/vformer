import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..utils import ATTENTION_REGISTRY
from .vanilla import VanillaSelfAttention


@ATTENTION_REGISTRY.register()
class GatedPositionalSelfAttention(VanillaSelfAttention):
    """
    Implementation of the Gated Positional Self-Attention from the paper: "ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases"

    Parameters
    ----------
    dim: int
        Dimension of the embedding
    num_heads: int
        Number of the attention heads, default is 8
    head_dim: int
        Dimension of each head, default is 64
    p_dropout: float
        Dropout probability, default is 0.0
    """

    def __init__(self, dim, num_heads=8, head_dim=64, p_dropout=0):
        super().__init__(dim, num_heads, head_dim, p_dropout)

        self.gate = nn.Parameter(torch.ones([1, num_heads, 1, 1]))
        self.pos = nn.Linear(3, num_heads)
        self.n_prev = 0

    def rel_embedding(self, n):

        l = int(n ** 0.5)
        rel_indices_x = torch.arange(l).reshape(1, -1)
        rel_indices_y = torch.arange(l).reshape(-1, 1)
        indices = rel_indices_x - rel_indices_y
        rel_indices_x = indices.repeat(l, l)
        rel_indices_y = indices.repeat_interleave(l, dim=0).repeat_interleave(l, dim=1)
        rel_indices_d = (rel_indices_x ** 2 + rel_indices_y ** 2) ** 0.5
        self.rel_indices = torch.stack(
            [rel_indices_x, rel_indices_y, rel_indices_d], dim=-1
        )
        self.rel_indices = self.rel_indices.unsqueeze(0)
        self.rel_indices = self.rel_indices.to(self.gate.device)

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
        _, N, _ = x.shape

        if not self.n_prev == N:
            self.n_prev = N
            self.rel_embedding(N)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        pos_attn = self.pos(self.rel_indices)
        attn = self.attend(dots)

        pos_attn = rearrange(pos_attn, "b n d h -> b h n d")
        pos_attn = F.softmax(pos_attn, dim=-1)
        attn = (1 - torch.sigmoid(self.gate)) * attn + torch.sigmoid(
            self.gate
        ) * pos_attn

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)
