import torch.nn as nn

from ..attention import VanillaSelfAttention
from ..functional import PreNorm
from .nn import FeedForward


class VanillaEncoder(nn.Module):
    def __init__(self, latent_dim, depth, heads, dim_head, mlp_dim, p_dropout=0.0):
        super().__init__()

        self.encoder = nn.ModuleList([])

        for _ in range(depth):
            self.encoder.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            latent_dim,
                            VanillaSelfAttention(
                                latent_dim,
                                heads=heads,
                                dim_head=dim_head,
                                p_dropout=p_dropout,
                            ),
                        ),
                        PreNorm(
                            latent_dim,
                            FeedForward(latent_dim, mlp_dim, p_dropout=p_dropout),
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.encoder:
            x = attn(x) + x
            x = ff(x) + x

        return x
