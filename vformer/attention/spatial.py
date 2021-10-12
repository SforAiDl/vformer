import torch
import torch.nn as nn

from ..functional import PreNorm


class SpatialAttention(nn.Module):
    """
    Spatial Reduction Attention

    Parameters
    -----------
    dim: int
    num_heads: int
    dim_head: int
    sr_ratio :int
    qkv_bias : bool
    qk_scale : float, optional
    attn_drop : float
    proj_drop :float
    linear : bool
    act_fn : activation function
    """

    def __init__(
        self,
        dim,
        num_heads,
        dim_head,
        sr_ratio=1,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        linear=False,
        act_fn=nn.GELU,
    ):
        super(SpatialAttention, self).__init__()
        self.num_heads = num_heads
        inner_dim = dim_head * num_heads

        self.sr_ratio = sr_ratio
        self.scale = qk_scale
        self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, inner_dim * 2, bias=qkv_bias)

        self.attn = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(p=attn_drop))

        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(p=proj_drop))
        self.linear = linear
        self.sr_ratio = sr_ratio
        self.norm = PreNorm(dim=dim, fn=act_fn() if linear else nn.Identity())
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.norm(self.sr(x_).reshape(B, C, -1).permute(0, 2, 1))
                kv = (
                    self.kv(x_)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
            else:
                kv = (
                    self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.norm(self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1))
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.to_out(x)
