import torch
import torch.nn as nn

from ..attention import CrossAttention
from ..utils import ENCODER_REGISTRY
from .vanilla import VanillaEncoder


@ENCODER_REGISTRY.register()
class CrossEncoder(nn.Module):
    """

    Parameters
    ----------
    embedding_dim_s : int
        Dimension of the embedding of smaller patches, default is 1024
    embedding_dim_l : int
        Dimension of the embedding of larger patches, default is 1024
    attn_heads_s : int
        Number of self-attention heads for the smaller patches, default is 16
    attn_heads_l : int
        Number of self-attention heads for the larger patches, default is 16
    cross_head_s : int
        Number of cross-attention heads for the smaller patches, default is 8
    cross_head_l : int
        Number of cross-attention heads for the larger patches, default is 8
    head_dim_s : int
        Dimension of the head of the attention for the smaller patches, default is 64
    head_dim_l : int
        Dimension of the head of the attention for the larger patches, default is 64
    cross_dim_head_s : int
        Dimension of the head of the cross-attention for the smaller patches, default is 64
    cross_dim_head_l : int
        Dimension of the head of the cross-attention for the larger patches, default is 64
    depth_s : int
        Number of self-attention layers in encoder for the smaller patches, default is 6
    depth_l : int
        Number of self-attention layers in encoder for the larger patches, default is 6
    mlp_dim_s : int
        Dimension of the hidden layer in the feed-forward layer for the smaller patches, default is 2048
    mlp_dim_l : int
        Dimension of the hidden layer in the feed-forward layer for the larger patches, default is 2048
    p_dropout_s : float
        Dropout probability for the smaller patches, default is 0.0
    p_dropout_l : float
        Dropout probability for the larger patches, default is 0.0
    """

    def __init__(
        self,
        embedding_dim_s=1024,
        embedding_dim_l=1024,
        attn_heads_s=16,
        attn_heads_l=16,
        cross_head_s=8,
        cross_head_l=8,
        head_dim_s=64,
        head_dim_l=64,
        cross_dim_head_s=64,
        cross_dim_head_l=64,
        depth_s=6,
        depth_l=6,
        mlp_dim_s=2048,
        mlp_dim_l=2048,
        p_dropout_s=0.0,
        p_dropout_l=0.0,
    ):
        super().__init__()

        self.s = VanillaEncoder(
            embedding_dim_s,
            depth_s,
            attn_heads_s,
            head_dim_s,
            mlp_dim_s,
            p_dropout_s,
        )
        self.l = VanillaEncoder(
            embedding_dim_l,
            depth_l,
            attn_heads_l,
            head_dim_l,
            mlp_dim_l,
            p_dropout_l,
        )
        self.attend_s = CrossAttention(
            embedding_dim_s, embedding_dim_l, cross_head_s, cross_dim_head_s
        )
        self.attend_l = CrossAttention(
            embedding_dim_l, embedding_dim_s, cross_head_l, cross_dim_head_l
        )

    def forward(self, emb_s, emb_l):

        emb_s = self.s(emb_s)
        emb_l = self.l(emb_l)
        s_cls, s_patches = (lambda t: (t[:, 0:1, :], t[:, 1:, :]))(emb_s)
        l_cls, l_patches = (lambda t: (t[:, 0:1, :], t[:, 1:, :]))(emb_l)
        s_cls = self.attend_s(s_cls, l_patches)
        l_cls = self.attend_l(l_cls, s_patches)
        emb_l = torch.cat([l_cls, l_patches], dim=1)
        emb_s = torch.cat([s_cls, s_patches], dim=1)

        return emb_s, emb_l
