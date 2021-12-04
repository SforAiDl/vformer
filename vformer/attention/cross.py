import torch
import torch.nn as nn
from einops import rearrange


class _Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        if not in_dim == out_dim:
            self.l1 = nn.Linear(in_dim, out_dim)
        else:
            self.l1 = nn.Identity()

    def forward(self, x):
        return self.l1(x)


class CrossAttention(nn.Module):
    """
    Cross-Attention Fusion

    Parameters
    ----------
    cls_dim: int
        Dimension of cls token embedding
    patch_dim: int
        Dimension of patch token embeddings cls token to be fused with
    heads: int
        Number of cross-attention heads
    dim_head: int
        Dimension of each head

    """

    def __init__(self, cls_dim, patch_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.fl = _Projection(cls_dim, patch_dim)
        self.gl = _Projection(patch_dim, cls_dim)
        self.to_k = nn.Linear(patch_dim, inner_dim)
        self.to_v = nn.Linear(patch_dim, inner_dim)
        self.to_q = nn.Linear(patch_dim, inner_dim)
        self.cls_project = _Projection(inner_dim, patch_dim)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, cls, patches):
        cls = self.fl(cls)
        x = torch.cat([cls, patches], dim=-2)
        q = self.to_q(cls)
        k = self.to_k(x)
        v = self.to_v(x)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        k = torch.transpose(k, -2, -1)
        attention = (q @ k) * self.scale
        attention = self.attend(attention)
        attention_value = attention @ v
        attention_value = rearrange(attention_value, "b h n d -> b n (h d)")
        attention_value = self.cls_project(attention_value)
        ycls = cls + attention_value
        ycls = self.gl(ycls)
        return ycls
