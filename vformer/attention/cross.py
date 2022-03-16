import torch
import torch.nn as nn
from einops import rearrange, repeat

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
class CrossAttentionWithClsToken(nn.Module):
    """
    Cross-Attention with Cls Token

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
        h = self.num_heads
        cls = self.fl(cls)

        x = torch.cat([cls, patches], dim=-2)
        q = self.to_q(cls)
        k = self.to_k(x)
        v = self.to_v(x)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = self.attend(attention)
        attention_value = attention @ v
        attention_value = rearrange(attention_value, "b h n d -> b n (h d)")
        attention_value = self.cls_project(attention_value)

        ycls = cls + attention_value
        ycls = self.gl(ycls)

        return ycls


@ATTENTION_REGISTRY.register()
class CrossAttention(nn.Module):
    """
    Cross-Attention

    Parameters
    ----------
    query_dim: int
        Dimension of query array
    context_dim: int
        Dimension of context array
    num_heads: int
        Number of cross-attention heads
    head_dim: int
        Dimension of each head

    """

    def __init__(self, query_dim, context_dim, num_heads=8, head_dim=64):
        super().__init__()

        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(context_dim, inner_dim)
        self.to_v = nn.Linear(context_dim, inner_dim)
        self.to_out = _Projection(inner_dim, query_dim)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x, context, mask=None):
        h = self.num_heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        attention = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(attention.dtype).max
            mask = repeat(mask, "b j -> b h () j", h=h)
            attention.masked_fill_(~mask, max_neg_value)

        attention = self.attend(attention)
        attention_value = attention @ v
        attention_value = rearrange(attention_value, "b h n d -> b n (h d)")

        return self.to_out(attention_value)
