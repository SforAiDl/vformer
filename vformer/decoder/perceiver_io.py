import torch.nn as nn
from einops import rearrange, repeat

from ..attention.cross import CrossAttention
from ..encoder.nn import FeedForward
from ..functional import PreNorm
from ..utils import DECODER_REGISTRY


@DECODER_REGISTRY.register()
class PerceiverIODecoder(nn.Module):
    """
    Implementation of the Perceiver IO Decoder

    Parameters
    ----------
    dim: int
        Size of sequence to be encoded
    latent_dim: int
        Dimension of latent array
    queries_dim: int
        Dimension of queries array
    num_latents: int
        Number of latent arrays
    num_cross_heads: int
        Number of heads for cross attention
    cross_head_dim: int
        Dimension of cross attention head
    logits_dim: int, optional
        Dimension of output logits
    decoder_ff: bool
        Whether to include a feed forward layer for the decoder attention block
    """

    def __init__(
        self,
        dim=32,
        latent_dim=512,
        queries_dim=32,
        num_cross_heads=1,
        cross_head_dim=64,
        logits_dim=None,
        decoder_ff=False,
    ):
        super().__init__()

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            CrossAttention(
                queries_dim,
                latent_dim,
                num_heads=num_cross_heads,
                head_dim=cross_head_dim,
            ),
            context_dim=latent_dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_logits = (
            nn.Linear(queries_dim, logits_dim)
            if logits_dim is not None
            else nn.Identity()
        )

    def forward(self, x, mask=None, queries=None):
        b, *_, device = *x.shape, x.device

        if queries is None:
            return x

        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=b)
        latents = self.decoder_cross_attn(queries, context=x)

        if self.decoder_ff is not None:
            latents = latents + self.decoder_ff(latents)
        return self.to_logits(latents)
