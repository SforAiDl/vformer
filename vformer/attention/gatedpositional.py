import torch
import torch.nn as nn
from einops import rearrange

from .vanilla import VanillaSelfAttention


class GatedPositionalSelfAttention(VanillaSelfAttention):
    def __init__(self, dim, num_heads=8, head_dim=64, p_dropout=0):
        super().__init__(dim, num_heads, head_dim, p_dropout)

        self.gate = nn.Parameter(torch.ones([num_heads]))
        self.flag = True

    def rel_embedding(self, n):
        rel_indices_x = torch.arange(n ** 0.5).repeat(1, n)
        print(rel_indices_x.shape)

    def forward(self, x):
        # x = b,n,emb
        _, N, _ = x.shape
        qkv = super().to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=super().num_heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * super().scale

        attn = self.attend(dots)


a = GatedPositionalSelfAttention(64, 8, 64)
a.rel_embedding(16)
