import torch.nn as nn
from timm.models.layers import DropPath

from ..attention import GatedPositionalSelfAttention
from ..functional import PreNorm
from ..utils import ENCODER_REGISTRY
from .nn import FeedForward
from .vanilla import VanillaEncoder


@ENCODER_REGISTRY.register()
class ConViTEncoder(VanillaEncoder):
    """

    Parameters
    ----------
    embedding_dim: int
        Dimension of the embedding
    depth: int
        Number of self-attention layers
    num_heads: int
        Number of the attention heads
    head_dim: int
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
        embedding_dim,
        depth,
        num_heads,
        head_dim,
        mlp_dim,
        p_dropout=0,
        attn_dropout=0,
        drop_path_rate=0,
    ):
        super().__init__(
            embedding_dim,
            depth,
            num_heads,
            head_dim,
            mlp_dim,
            p_dropout,
            attn_dropout,
            drop_path_rate,
        )
        self.encoder = nn.ModuleList([])

        for _ in range(depth):
            self.encoder.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim=embedding_dim,
                            fn=GatedPositionalSelfAttention(
                                dim=embedding_dim,
                                num_heads=num_heads,
                                head_dim=head_dim,
                                p_dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(
                            dim=embedding_dim,
                            fn=FeedForward(
                                dim=embedding_dim,
                                hidden_dim=mlp_dim,
                                p_dropout=p_dropout,
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
