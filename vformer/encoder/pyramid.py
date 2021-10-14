import torch
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
                        DropPath(drop_prob=drop_path[i])
                        if drop_path > 0.0
                        else nn.Identity(),
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
                DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
            )

    def forward(self, x, H, W):
        for attn, drop_path, ff in self.encoder:
            x = x + drop_path(attn(x, H, W))
            x = x + drop_path(ff(x, H, W))
        return x
