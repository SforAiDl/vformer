import torch
import torch.nn as nn

from ...decoder import MLPDecoder
from ...attention import WindowAttention, WindowAttentionGlobal

from ..utils import (
    ENCODER_REGISTRY,
    window_partition,
    window_reverse,
)

@ENCODER_REGISTRY.register()
class GCViTBlock(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=WindowAttentionGlobal,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 ):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)

        self.attn = attention(dim,
                              num_heads=num_heads,
                              window_size=window_size,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLPDecoder(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

        self.num_windows = int((input_resolution // window_size) * (input_resolution // window_size))

    def forward(self, x, q_global):
            B, H, W, C = x.shape
            shortcut = x
            x = self.norm1(x)
            x_windows = window_partition(x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            attn_windows = self.attn(x_windows, q_global)
            x = window_reverse(attn_windows, self.window_size, H, W)
            x = shortcut + self.drop_path(self.gamma1 * x)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

@ENCODER_REGISTRY.register()
class GCViTLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 input_resolution,
                 num_heads,
                 window_size,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            GCViTBlock(dim=dim,
                       num_heads=num_heads,
                       window_size=window_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias,
                       qk_scale=qk_scale,
                       attention=WindowAttention if (i % 2 == 0) else WindowAttentionGlobal,
                       drop=drop,
                       attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer,
                       layer_scale=layer_scale,
                       input_resolution=input_resolution)
            for i in range(depth)])

        self.downsample = None if not downsample else ReduceSize(dim=dim, norm_layer=norm_layer)

        if input_resolution == 56:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == 28:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == 14:

            if window_size == 14:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=True)
                )

            elif window_size == 7:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=False)
                )

        elif input_resolution == 7:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=True)
            )

        self.dim = dim
        self.resolution = input_resolution

    def forward(self, x):
        q_global = self.to_q_global(x.view(-1,
                                           self.dim,
                                           self.resolution,
                                           self.resolution))
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is None:
            return x
        return self.downsample(x)
