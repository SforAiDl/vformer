import torch.nn as nn
from timm.models.layers import DropPath

from ..attention import VanillaSelfAttention
from ..functional import PreNorm
from ..utils import ENCODER_REGISTRY
from .nn import FeedForward


@ENCODER_REGISTRY.register()
class VanillaEncoder(nn.Module):
    """

    Parameters
    ----------
    latent_dim: int
        Dimension of the embedding
    depth: int
        Number of self-attention layers
    heads: int
        Number of the attention heads
    dim_head: int
        Dimension of each head
    mlp_dim: int
        Dimension of the hidden layer in the feed-forward layer
    p_dropout: float
        Dropout Probability
    attn_dropout: float
        Dropout Probability
    drop_path_rate: float
        Stochastic drop path rate
    """

    def __init__(
        self,
        latent_dim,
        depth,
        num_heads,
        dim_head,
        mlp_dim,
        p_dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.encoder = nn.ModuleList([])
        for _ in range(depth):
            self.encoder.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim=latent_dim,
                            fn=VanillaSelfAttention(
                                dim=latent_dim,
                                num_heads=num_heads,
                                head_dim=dim_head,
                                p_dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(
                            dim=latent_dim,
                            fn=FeedForward(
                                dim=latent_dim, hidden_dim=mlp_dim, p_dropout=p_dropout
                            ),
                        ),
                    ]
                )
            )
        self.drop_path = (
            DropPath(drop_prob=drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x):

        for attn, ff in self.encoder:
            x = attn(x) + x
            x = self.drop_path(ff(x)) + x

        return x
