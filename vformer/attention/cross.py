import torch
import torch.nn as nn
from einops import rearrange

from ..utils import ATTENTION_REGISTRY


class _Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if not in_dim == out_dim:
            self.l1 = nn.Linear(in_dim, out_dim)
        else:
            self.l1 = nn.Identity()

    def forward(self, x):
        return self.l1(x)


@ATTENTION_REGISTRY.register()
class CrossAttention(nn.Module):
    """
    Cross-Attention Fusion

    Parameters
    ----------
    cls_dim: int
        Dimension of cls token embedding
    patch_dim: int
        Dimension of patch token embeddings cls token to be fused with
    num_heads: int
        Number of cross-attention heads
    head_dim: int
        Dimension of each head

    """

    def __init__(self, cls_dim, patch_dim, num_heads=8, head_dim=64):
        super().__init__()

        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
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
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        k = torch.transpose(k, -2, -1)

        attention = (q @ k) * self.scale
        attention = self.attend(attention)
        attention_value = attention @ v
        attention_value = rearrange(attention_value, "b h n d -> b n (h d)")
        attention_value = self.cls_project(attention_value)

        ycls = cls + attention_value
        ycls = self.gl(ycls)

        return ycls
