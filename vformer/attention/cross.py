import torch
import torch.nn as nn
from einops import rearrange, repeat

from ..utils import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register()
class CrossAttentionWithClsToken(nn.Module):
    """
    Cross-Attention with Cls Token introduced in Paper: CrossViT: `Cross-Attention Multi-Scale Vision Transformer for Image Classification <https://arxiv.org/abs/2103.14899>`_

    In Cross-Attention, cls token from one branch and patch token from another branch are fused together.

    Parameters
    -----------
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
        self.scale = head_dim**-0.5
        self.fl = (
            nn.Linear(cls_dim, patch_dim) if not cls_dim == patch_dim else nn.Identity()
        )
        self.gl = (
            nn.Linear(patch_dim, cls_dim) if not cls_dim == patch_dim else nn.Identity()
        )

        self.to_k = nn.Linear(patch_dim, inner_dim)
        self.to_v = nn.Linear(patch_dim, inner_dim)
        self.to_q = nn.Linear(patch_dim, inner_dim)
        self.cls_project = (
            nn.Linear(inner_dim, patch_dim) if inner_dim != patch_dim else nn.Identity()
        )

        self.attend = nn.Softmax(dim=-1)

    def forward(self, cls, patches):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        cls: torch.Tensor
            CLS token from one branch
        patch: torch.Tensor
            patch tokens from another branch
        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying cross attention on input tensor

        """
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
    This variant of Cross Attention is iteratively used in Perciever IO.

    In Cross-Attention, cls token from one branch and patch token from another branch are fused together.

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
        self.scale = head_dim**-0.5
        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(context_dim, inner_dim)
        self.to_v = nn.Linear(context_dim, inner_dim)
        self.to_out = (
            nn.Linear(inner_dim, query_dim)
            if not inner_dim == query_dim
            else nn.Identity()
        )
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
