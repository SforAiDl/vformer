import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

from ..attention.window import WindowAttention, WindowAttentionGlobal
from ..decoder import MLPDecoder
from ..utils import ENCODER_REGISTRY, window_partition, window_reverse


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ReduceSize(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, keep_dim=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False)
        self.norm2 = norm_layer(dim_out)
        self.norm1 = norm_layer(dim)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = x + self.conv(x)
        x = self.reduction(x).permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x


class FeatExtract(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x


class GlobalQueryGen(nn.Module):
    def __init__(self, dim, input_resolution, window_size, num_heads):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            num_heads: number of heads.
        For instance, repeating log(56/7) = 3 blocks, with input window dimension 56 and output window dimension 7 at
        down-sampling ratio 2. Please check Fig.5 of GC ViT paper for details.
        """

        super().__init__()
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
                self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=True))

            elif window_size == 7:
                self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=False))

        elif input_resolution == 7:
            self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=True))

        self.resolution = input_resolution
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = dim // self.num_heads

    def forward(self, x):
        x = self.to_q_global(x)
        x = x.permute(0, 2, 3, 1)
        B = x.shape[0]
        x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(
            0, 1, 3, 2, 4
        )
        return x


@ENCODER_REGISTRY.register()
class GCViTBlock(nn.Module):
    """
    GCViT block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"


    Parameters
    ----------
    dim: feature size dimension.
    input_resolution: input image resolution.
    num_heads: number of attention head.
    window_size: window size.
    mlp_ratio: MLP ratio.
    qkv_bias: bool argument for query, key, value learnable bias.
    qk_scale: bool argument to scaling query, key.
    drop: dropout rate.
    attn_drop: attention dropout rate.
    drop_path_rate: drop path rate.
    drop_path_mode: drop path mode.
    act_layer: activation function.
    attention: attention block type.
    norm_layer: normalization layer.
    layer_scale: layer scaling coefficient.

    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        drop_path_mode="batch",
        act_layer=nn.GELU,
        attention=WindowAttentionGlobal,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):

        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)

        self.attn = attention(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = (
            StochasticDepth(p=drop_path_rate, mode=drop_path_mode)
            if drop_path_rate > 0.0
            else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPDecoder(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

        self.num_windows = int(
            (input_resolution // window_size) * (input_resolution // window_size)
        )

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
    def __init__(
        self,
        dim,
        depth,
        input_resolution,
        num_heads,
        window_size,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GCViTBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attention=WindowAttention
                    if (i % 2 == 0)
                    else WindowAttentionGlobal,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path_rate=drop_path_rate[i]
                    if isinstance(drop_path_rate, list)
                    else drop_path_rate,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    input_resolution=input_resolution,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ReduceSize(dim=dim, norm_layer=norm_layer)
        )

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
                self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=True))

            elif window_size == 7:
                self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=False))

        elif input_resolution == 7:
            self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=True))

        self.dim = dim
        self.resolution = input_resolution

    def forward(self, x):
        q_global = self.to_q_global(
            x.view(-1, self.dim, self.resolution, self.resolution)
        )
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is None:
            return x
        return self.downsample(x)


class GCViTLayer(nn.Module):
    """
    GCViT layer based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"

    Parameters
    ----------
    dim: feature size dimension.
    depth: number of layers in each stage.
    input_resolution: input image resolution.
    window_size: window size in each stage.
    downsample: bool argument for down-sampling.
    mlp_ratio: MLP ratio.
    num_heads: number of heads in each stage.
    qkv_bias: bool argument for query, key, value learnable bias.
    qk_scale: bool argument to scaling query, key.
    drop: dropout rate.
    attn_drop: attention dropout rate.
    drop_path_rate: drop path rate.
    norm_layer: normalization layer.
    layer_scale: layer scaling coefficient.

    """

    def __init__(
        self,
        dim,
        depth,
        input_resolution,
        num_heads,
        window_size,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
    ):

        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GCViTBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attention=WindowAttention
                    if (i % 2 == 0)
                    else WindowAttentionGlobal,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_rate[i]
                    if isinstance(drop_path_rate, list)
                    else drop_path_rate,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    input_resolution=input_resolution,
                )
                for i in range(depth)
            ]
        )
        self.downsample = (
            None if not downsample else ReduceSize(dim=dim, norm_layer=norm_layer)
        )
        self.q_global_gen = GlobalQueryGen(
            dim, input_resolution, window_size, num_heads
        )

    def forward(self, x):
        q_global = self.q_global_gen(x.permute(0, 3, 1, 2))
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is None:
            return x
        return self.downsample(x)
