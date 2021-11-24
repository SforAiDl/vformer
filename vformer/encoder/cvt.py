import torch
import torch.nn as nn
from torch.nn import  Module,Linear,LayerNorm,Identity,Dropout
import torch.nn.functional as F
from ..attention import VanillaSelfAttention
from timm.models.layers import DropPath
from .nn import FeedForward

from ..functional.norm import PreNorm
class CVTEncoder(nn.Module):
    def __init__(self,d_model,nhead,p_dropout,hidden_dim=None,out_dim=None,drop_path_rate=0.):
        super(CVTEncoder, self).__init__()
        self.encoder= nn.ModuleList([])
        self.encoder.append(
            nn.ModuleList([
            PreNorm(
                dim=d_model,
                fn=VanillaSelfAttention(
                    dim=d_model,
                    heads=nhead,
                    p_dropout=p_dropout,
                )
            ),

            PreNorm(
                dim=d_model,
                fn=FeedForward(dim=d_model,hidden_dim=hidden_dim,out_dim=out_dim,p_dropout=p_dropout)
            )]
        ))
        self.norm= nn.LayerNorm(d_model)
        self.drop_path = (
            DropPath(drop_prob=drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )
    def forward(self,x):
        for attn,ff in self.encoder:
            x = attn(x) + x
            x=self.norm(x)
            x = ff(x) + x
        x=x+self.drop_path(x)
        return x

