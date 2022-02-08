import torch
import torch.nn as nn

from ..attention import VanillaSelfAttention
from ..functional.norm import PreNorm
from ..utils.registry import ENCODER_REGISTRY
from .nn import FeedForward


@ENCODER_REGISTRY.register()
class ViViTEncoderBlock(nn.Module):
    """For model 3 only """

    def __init__(
        self, dim, num_heads, head_dim, p_dropout, out_dim=None, hidden_dim=None
    ):
        super(ViViTEncoderBlock, self).__init__()

        self.temporal_attention = PreNorm(
            dim=dim, fn=VanillaSelfAttention(dim, num_heads, head_dim, p_dropout)
        )
        self.spatial_attention = PreNorm(
            dim=dim, fn=VanillaSelfAttention(dim, num_heads, head_dim, p_dropout)
        )

        self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim, out_dim=out_dim)

    def forward(self, x):

        b, n, s, d = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=1)  # 1×nt·nh·nw·d --> nt×nh·nw·d

        x = self.spatial_attention(x) + x

        x = x.reshape(b, n, s, d).transpose(1, 2)
        x = torch.flatten(x, start_dim=0, end_dim=1)  # nt×nh·nw·d --> nh·nw×nt·d

        x = self.temporal_attention(x) + x

        x = self.mlp(x) + x

        x = x.reshape(
            b, n, s, d
        )  # reshaping because this block is used for several depths in ViViTEncoder class and Next layer will expect the x in proper shape

        return x


@ENCODER_REGISTRY.register()
class ViViTEncoder(nn.Module):
    """model 3 only"""

    def __init__(
        self, dim, num_heads, head_dim, p_dropout, depth, out_dim=None, hidden_dim=None
    ):
        super(ViViTEncoder, self).__init__()
        self.encoder = nn.ModuleList()

        for _ in range(depth):
            self.encoder.append(
                ViViTEncoderBlock(
                    dim, num_heads, head_dim, p_dropout, out_dim, hidden_dim
                )
            )

    def forward(self, x):

        b = x.shape[0]

        for blk in self.encoder:
            x = blk(x)

        x = x.reshape(b, -1, x.shape[-1])

        return x
