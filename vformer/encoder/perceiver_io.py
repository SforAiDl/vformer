import torch
import torch.nn as nn
from einops import repeat

from ..attention.cross import CrossAttention
from ..attention.vanilla import VanillaSelfAttention
from ..encoder.nn import FeedForward
from ..functional import PreNorm
from ..utils import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class PerceiverIOEncoder(nn.Module):
    """
    Implementation of the Perceiver IO Encoder containing Iterative Cross Attention and Processor

    Parameters
    ----------
    dim: int
        Size of sequence to be encoded
    depth: int
        Depth of latent attention blocks
    latent_dim: int
        Dimension of latent array
    num_latents: int
        Number of latent arrays
    num_cross_heads: int
        Number of heads for cross attention
    num_latent_heads: int
        Number of heads for latent attention
    cross_head_dim: int
        Dimension of cross attention head
    latent_head_dim: int
        Dimension of latent attention head
    """

    def __init__(
        self,
        dim=32,
        depth=6,
        latent_dim=512,
        num_latents=512,
        num_cross_heads=1,
        num_latent_heads=8,
        cross_head_dim=64,
        latent_head_dim=64,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attn = PreNorm(
            latent_dim,
            CrossAttention(
                latent_dim, dim, num_heads=num_cross_heads, head_dim=cross_head_dim
            ),
            context_dim=dim,
        )
        self.cross_ff = PreNorm(latent_dim, FeedForward(latent_dim))

        get_latent_attn = VanillaSelfAttention(
            latent_dim, num_heads=num_latent_heads, head_dim=latent_head_dim
        )
        get_latent_ff = PreNorm(latent_dim, FeedForward(latent_dim))

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([get_latent_attn, get_latent_ff]))

    def forward(self, x, mask=None):
        b, *_, device = *x.shape, x.device

        inner_x = repeat(self.latents, "n d -> b n d", b=b)
        inner_x = self.cross_attn(inner_x, context=x, mask=mask) + inner_x
        inner_x = self.cross_ff(inner_x) + inner_x

        for self_attn, self_ff in self.layers:
            inner_x = self_attn(inner_x) + inner_x
            inner_x = self_ff(inner_x) + inner_x

        return inner_x
