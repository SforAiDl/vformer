import torch
from torch import nn

from timm.models.layers import DropPath
from .nn import FeedForward as Mlp
from ..attention import MultiScaleAttention
from ..utils import ENCODER_REGISTRY

@ENCODER_REGISTRY.register()
class MultiScaleBlock(nn.Module):
    """
    Multiscale Attention Block
    Parameters:
    -----------
    dim: int
         Dimension of the embedding
    dim_out: int
             Output dimension of the embedding
    num_heads: int
               Number of attention heads
    mlp_ratio: float, optional
               Ratio of hidden dimension to input dimension for MLP
    qkv_bias :bool, optional
              If True, add a learnable bias to query, key, value.
    qk_scale: float, optional
              Override default qk scale of head_dim ** -0.5 if set
    drop_rate: float, optional
               Dropout rate
    drop_path: float, optional
               Dropout rate for dropping paths in mlp
    norm_layer= nn.Module, optional
                Normalization function
    up_rate= float, optional
             Ratio of output dimension to input dimension for MLP
    kernel_q: tuple of int, optional
              Kernel size of query
    kernel_kv: tuple of int, optional
               Kernel size of key and value
    stride_q: tuple of int, optional
              Kernel size of query
    stride_kv: tuple of int, optional
               Kernel size of key and value
    norm_layer: nn.Module, optional
                Normalization function
    mode: str, optional
          Pooling function to be used. Options include `conv`, `avg`, and `max'
    has_cls_embed: bool, optional
                   Set to true if classification embeddding is provided
    pool_first: bool, optional
                Set to True to perform pool before projection
    """

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            out_dim=mlp_dim_out,
            p_dropout=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(kernel_skip) > 0
            else None
        )

    def forward(self, x, thw_shape):
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new
