import torch.nn as nn
from timm.models.layers import DropPath

from ..attention import SpatialAttention
from ..functional import PreNorm
from .pvtfeedforward import PVTFeedForward


class PVTEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        depth,
        qkv_bias,
        qk_scale,
        drop,
        attn_drop,
        drop_path,
        act_layer,
        norm_layer,
        sr_ratio,
        linear=False,
    ):
        super(PVTEncoder, self).__init__()
        self.encoder = nn.ModuleList([])
        for i in range(depth):
            self.encoder.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim=dim,
                            fn=SpatialAttention(
                                dim=dim,
                                num_heads=num_heads,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                attn_drop=attn_drop,
                                proj_drop=drop,
                                sr_ratio=sr_ratio,
                                linear=linear,
                            ),
                        ),
                        PreNorm(
                            dim=dim,
                            fn=PVTFeedForward(
                                dim=dim,
                                hidden_dim=int(dim * mlp_ratio),
                                act_layer=act_layer,
                                drop=drop,
                                linear=linear,
                            ),
                        ),
                    ]
                )
            )
            self.drop_path = (
                DropPath(drop_prob=drop_path[i])
                if drop_path[i] > 0.0
                else nn.Identity()
            )

    def forward(self, x, **kwargs):
        for prenorm_attn, prenorm_ff in self.encoder:
            x = x + self.drop_path(prenorm_attn(x, **kwargs))
            x = x + self.drop_path(prenorm_ff(x, **kwargs))
        return x
