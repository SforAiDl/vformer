import torch
import torch.nn as nn

from ...decoder import PerceiverIODecoder
from ...encoder import PerceiverIOEncoder
from ...utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class PerceiverIO(nn.Module):
    """
    Implementation of `Perceiver IO: A General Architecture for Structured Inputs & Outputs <https://arxiv.org/abs/2107.14795>`_

    Code Implementation based on:
    https://github.com/lucidrains/perceiver-pytorch

    Parameters
    -----------
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
    queries_dim: int
        Dimension of queries array
    logits_dim: int, optional
        Dimension of output logits
    decoder_ff: bool
        Whether to include a feed forward layer for the decoder attention block
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
        queries_dim=32,
        logits_dim=None,
        decoder_ff=False,
    ):
        super().__init__()
        self.encoder = PerceiverIOEncoder(
            dim=dim,
            depth=depth,
            latent_dim=latent_dim,
            num_latents=num_latents,
            num_cross_heads=num_cross_heads,
            num_latent_heads=num_latent_heads,
            cross_head_dim=cross_head_dim,
            latent_head_dim=latent_head_dim,
        )

        self.decoder = PerceiverIODecoder(
            dim=dim,
            latent_dim=latent_dim,
            queries_dim=queries_dim,
            num_cross_heads=num_cross_heads,
            cross_head_dim=cross_head_dim,
            logits_dim=logits_dim,
            decoder_ff=decoder_ff,
        )

    def forward(self, x, queries):

        out = self.encoder(x)
        out = self.decoder(out, queries=queries)

        return out
