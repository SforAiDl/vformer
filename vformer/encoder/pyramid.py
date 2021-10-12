import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .nn import FeedForward
from ..attention import SpatialAttention
from ..functional import PreNorm

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
        dwconv=
        for _ in range(depth):
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
                        DropPath(drop_prob=drop_path) if drop_path>0. else nn.Identity(),

                        PreNorm(
                            dim=dim,
                            fn=FeedForward(dim=dim,
                                           hidden_dim=int(dim *mlp_ratio ),
                                           p_dropout=drop,
                                           fn=dwconv)
                        )
                    ]
                )
            )
    def forward(self,x):
        pass
