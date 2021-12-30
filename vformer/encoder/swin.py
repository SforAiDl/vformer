import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

from ..attention.window import WindowAttention
from ..utils import (
    ENCODER_REGISTRY,
    create_mask,
    cyclicshift,
    pair,
    window_partition,
    window_reverse,
)
from .nn import FeedForward


@ENCODER_REGISTRY.register()
class SwinEncoderBlock(nn.Module):
    """

    Parameters
    ----------
    dim: int
        Number of the input channels
    input_resolution: int or tuple[int]
        Input resolution of patches
    num_heads: int
        Number of attention heads
    window_size: int
        Window size
    shift_size: int
        Shift size for Shifted Window Masked Self Attention (SW_MSA)
    mlp_ratio: float
        Ratio of MLP hidden dimension to embedding dimension
    qkv_bias: bool, default= True
        Whether to add a bias vector to the q,k, and v matrices
    qk_scale: float, Optional

    p_dropout: float
        Dropout rate
    attn_dropout: float
        Dropout rate
    drop_path_rate: float
        Stochastic depth rate
    norm_layer:nn.Module
        Normalization layer, default is `nn.LayerNorm`

    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        p_dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super(SwinEncoderBlock, self).__init__()

        self.dim = dim
        self.input_resolution = pair(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = shift_size
        hidden_dim = int(dim * mlp_ratio)

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert (
            0 <= self.shift_size < window_size
        ), "shift size must range from 0 to window size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=pair(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_dropout=attn_dropout,
            proj_dropout=p_dropout,
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim, p_dropout=p_dropout)

        if self.shift_size > 0:
            attn_mask = create_mask(
                self.window_size,
                self.shift_size,
                H=self.input_resolution[0],
                W=self.input_resolution[1],
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor

        Returns
        ----------
        torch.Tensor
            Returns output tensor

        """

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input tensor shape not compatible"

        skip_connection = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = cyclicshift(x, shift_size=-self.shift_size)
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size).view(
            -1, self.window_size * self.window_size, C
        )

        attn_windows = self.attn(x_windows, mask=self.attn_mask).view(
            -1, self.window_size, self.window_size, C
        )

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = cyclicshift(shifted_x, shift_size=self.shift_size).view(B, H * W, C)
        else:
            x = shifted_x.view(B, H * W, C)

        x = skip_connection + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


@ENCODER_REGISTRY.register()
class SwinEncoder(nn.Module):
    """
    dim: int
        Number of input channels.
    input_resolution: tuple[int]
        Input resolution.
    depth: int
        Number of blocks.
    num_heads: int
        Number of attention heads.
    window_size: int
        Local window size.
    mlp_ratio: float
        Ratio of MLP hidden dim to embedding dim.
    qkv_bias: bool, default is True
       Whether to add a bias vector to the q,k, and v matrices
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 in Window Attention if set
    p_dropout: float,
        Dropout rate.
    attn_dropout: float, optional
        Attention dropout rate
    drop_path_rate: float or tuple[float]
        Stochastic depth rate.
    norm_layer: nn.Module
        Normalization layer. default is nn.LayerNorm
    downsample: nn.Module, optional
        Downsample layer(like PatchMerging) at the end of the layer, default is None

    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_scale=None,
        p_dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super(SwinEncoder, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                SwinEncoderBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qkv_scale,
                    p_dropout=p_dropout,
                    attn_dropout=attn_dropout,
                    drop_path_rate=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor

        Returns
        ----------
        torch.Tensor
            Returns output tensor

        """

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
