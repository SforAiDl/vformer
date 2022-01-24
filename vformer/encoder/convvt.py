import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from ..attention.convvt import ConvVTAttention
from ..encoder.nn import FeedForward
from ..utils import ENCODER_REGISTRY
from .embedding.convvt import ConvEmbedding


@ENCODER_REGISTRY.register()
class ConvVTStage(nn.Module):
    """
    Implementation of a Stage in CVT

    Parameters
    ----------
    patch_size: int
        Size of patch, default is 16
    patch_stride: int
        Stride of patch, default is 4
    patch_padding: int
        Padding for patch, default is 0
    in_channels:int
        Number of input channels in image, default is 3
    img_size: int
        Size of the image, default is 224
    embedding_dim: int
        Embedding dimensions, default is 64
    depth: int
        Number of CVT Attention blocks in each stage, default is 1
    num_heads: int
        Number of heads in attention, default is 6
    mlp_ratio: float
        Feature dimension expansion ratio in MLP, default is 4.0
    p_dropout: float
        Probability of dropout in MLP, default is 0.0
    attn_dropout: float
        Probability of dropout in attention, default is 0.0
    drop_path_rate: float
        Probability for droppath, default is 0.0
    with_cls_token: bool
        Whether to include classification token, default is False
    kernel_size: int
        Size of kernel, default is 3
    padding_q: int
        Size of padding in q, default is 1
    padding_kv: int
        Size of padding in kv, default is 2
    stride_kv: int
        Stride in kv, default is 2
    stride_q: int
        Stride in q, default is 1
    init: str ('trunc_norm' or 'xavier')
        Initialization method, default is 'trunc_norm'
    """

    def __init__(
        self,
        patch_size=7,
        patch_stride=4,
        patch_padding=0,
        in_channels=3,
        embedding_dim=64,
        depth=1,
        p_dropout=0.0,
        drop_path_rate=0.0,
        with_cls_token=False,
        init="trunc_norm",
        **kwargs
    ):
        super().__init__()

        self.patch_embed = ConvEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            stride=patch_stride,
            padding=patch_padding,
        )

        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=p_dropout)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                ConvVTBlock(
                    dim_in=embedding_dim,
                    dim_out=embedding_dim,
                    p_dropout=p_dropout,
                    with_cls_token=with_cls_token,
                    drop_path=dpr[j],
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)

        if init == "xavier":
            self.apply(self._init_weights_xavier)
        elif init == "trunc_norm":
            self.apply(self._init_weights_trunc_normal)
        else:
            raise ValueError("Init method {} not found".format(init))

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, "b c h w -> b (h w) c")

        cls_tokens = None
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H * W], 1)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x, cls_tokens


class ConvVTBlock(nn.Module):
    """
    Implementation of a Attention MLP block in CVT

    Parameters:
    ------------
    dim_in: int
        Input dimensions
    dim_out: int
        Output dimensions
    num_heads: int
        Number of heads in attention
    img_size: int
        Size of image
    mlp_ratio: float
        Feature dimension expansion ratio in MLP, default is 4.
    p_dropout: float
        Probability of dropout in MLP, default is 0.0
    attn_dropout: float
        Probability of dropout in attention, default is 0.0
    drop_path: float
        Probability of droppath, default is 0.0
    with_cls_token: bool
        Whether to include classification token, default is False
    """

    def __init__(
        self, dim_in, dim_out, mlp_ratio=4.0, p_dropout=0.0, drop_path=0.0, **kwargs
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim_in)
        self.attn = ConvVTAttention(dim_in, dim_out, **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = FeedForward(
            dim=dim_out, hidden_dim=dim_mlp_hidden, p_dropout=p_dropout
        )

    def forward(self, x):

        res = x

        x = self.norm1(x)
        attn = self.attn(x)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
