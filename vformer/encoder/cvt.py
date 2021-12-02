import torch.nn as nn
from timm.models.layers import DropPath

from ..attention import VanillaSelfAttention
from ..functional.norm import PreNorm
from .nn import FeedForward


class CVTEncoderBlock(nn.Module):
    """
    parameters:
    -----------
    dim:int
        Dimension of the input tensor
    p_dropout: float
        Dropout probability
    attn_dropout: float
        Dropout probability
    hidden_dim: int, optional
        Dimension of the hidden layer
    out_dim:int, optional
        Dimension of the output
    drop_path_rate:float
        Stochastic drop path rate

    """

    def __init__(
        self,
        dim,
        num_head,
        p_dropout,
        attn_dropout,
        hidden_dim=None,
        out_dim=None,
        drop_path_rate=0.0,
    ):
        super(CVTEncoderBlock, self).__init__()
        self.encoder = nn.ModuleList([])
        self.encoder.append(
            nn.ModuleList(
                [
                    PreNorm(
                        dim=dim,
                        fn=VanillaSelfAttention(
                            dim=dim,
                            heads=num_head,
                            p_dropout=attn_dropout,
                        ),
                    ),
                    PreNorm(
                        dim=dim,
                        fn=FeedForward(
                            dim=dim,
                            hidden_dim=hidden_dim,
                            out_dim=out_dim,
                            p_dropout=p_dropout,
                        ),
                    ),
                ]
            )
        )
        self.norm = nn.LayerNorm(dim)
        self.drop_path = (
            DropPath(drop_prob=drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x):
        for attn, ff in self.encoder:
            x = attn(x) + x
            x = self.norm(x)
            x = ff(x) + x
        x = x + self.drop_path(x)
        return x
