import torch.nn as nn

from ..attention import VanillaSelfAttention
from ..functional import PreNorm
from .nn import FeedForward


class VanillaEncoder(nn.Module):
    """
    class VanillaEncoder:
    Inputs-
    --------------
    latent_dim: embeding dimension
    depth: number of feedforward and attention blocks in encoding
    heads: number of heads in self attention block
    dim_head: head dimensions
    mlp_dim: dimension of hidden linear layer for the feedforward network
    p_dropout: probability for the dropout layer

    Forward method returns:
    vector passed through Transformer encoder architecture  for f'{depth}' number of times
    """
    def __init__(self, latent_dim:int, depth:int, heads:int, dim_head:int, mlp_dim:int, p_dropout:float=0.0):
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
