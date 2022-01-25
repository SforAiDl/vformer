from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..utils import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register()
class ConvVTAttention(nn.Module):
    """
    Attention with Convolutional Projection

    Parameters:
    ------------
    dim_in: int
        Dimension of input tensor
    dim_out: int
        Dimension of output tensor
    num_heads: int
        Number of heads in attention
    img_size: int
        Size of image
    attn_dropout: float
        Probability of dropout in attention
    proj_dropout: float
        Probability of dropout in convolution projection
    method: str ('dw_bn' for depth-wise convolution and batch norm, 'avg' for average pooling)
        Method of projection
    kernel_size: int
        Size of kernel
    stride_kv: int
        Size of stride for key value
    stride_q: int
        Size of stride for query
    padding_kv: int
        Padding for key value
    padding_q: int
        Padding for query
    with_cls_token: bool
        Whether to include classification token
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        img_size,
        attn_dropout=0.0,
        proj_dropout=0.0,
        method="dw_bn",
        kernel_size=3,
        stride_kv=1,
        stride_q=1,
        padding_kv=1,
        padding_q=1,
        with_cls_token=False,
    ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.with_cls_token = with_cls_token
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.h, self.w = img_size, img_size
        self.conv_proj_q = self._build_projection(
            dim_in, kernel_size, padding_q, stride_q, method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, kernel_size, padding_kv, stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, kernel_size, padding_kv, stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=False)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=False)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=False)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_dropout)

    def _build_projection(self, dim_in, kernel_size, padding, stride, method):
        if method == "dw_bn":
            proj = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                dim_in,
                                dim_in,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=False,
                                groups=dim_in,
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(dim_in)),
                    ]
                )
            )
        elif method == "avg":
            proj = nn.AvgPool2d(
                kernel_size=kernel_size, padding=padding, stride=stride, ceil_mode=True
            )
        else:
            raise ValueError("Unknown method ({})".format(method))

        return proj

    def forward_conv(self, x):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, self.h * self.w], 1)

        x = rearrange(x, "b (h w) c -> b c h w", h=self.h, w=self.w)

        q = self.conv_proj_q(x)
        q = rearrange(q, "b c h w -> b (h w) c")

        k = self.conv_proj_k(x)
        k = rearrange(k, "b c h w -> b (h w) c")

        v = self.conv_proj_v(x)
        v = rearrange(v, "b c h w -> b (h w) c")

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)

        q = rearrange(self.proj_q(q), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.proj_k(k), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.proj_v(v), "b t (h d) -> b h t d", h=self.num_heads)

        attn_score = torch.einsum("bhlk,bhtk->bhlt", [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum("bhlt,bhtv->bhlv", [attn, v])
        x = rearrange(x, "b h t d -> b t (h d)")

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
