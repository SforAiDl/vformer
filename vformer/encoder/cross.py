import torch
import torch.nn as nn

from vformer.attention import CrossAttention
from vformer.encoder.vanilla import VanillaEncoder


class CrossEncoder(nn.Module):
    """
    Parameters
    ----------
    latent_dim_s : int
        Dimension of the embedding of smaller patches
    latent_dim_l : int
        Dimension of the embedding of larger patches
    attn_heads_s : int
        Number of self-attention heads for the smaller patches
    attn_heads_l : int
        Number of self-attention heads for the larger patches
    cross_head_s : int
        Number of cross-attention heads for the smaller patches
    cross_head_l : int
        Number of cross-attention heads for the larger patches
    dim_head_s : int
        Dimension of the head of the attention for the smaller patches
    dim_head_l : int
        Dimension of the head of the attention for the larger patches
    cross_dim_head_s : int
        Dimension of the head of the cross-attention for the smaller patches
    cross_dim_head_l : int
        Dimension of the head of the cross-attention for the larger patches
    depth_s : int
        Number of self-attention layers in encoder for the smaller patches
    depth_l : int
        Number of self-attention layers in encoder for the larger patches
    mlp_dim_s : int
        Dimension of the hidden layer in the feed-forward layer for the smaller patches
    mlp_dim_l : int
        Dimension of the hidden layer in the feed-forward layer for the larger patches
    p_dropout_s : float
        Dropout probability for the smaller patches
    p_dropout_l : float
        Dropout probability for the larger patches
    """

    def __init__(
        self,
        latent_dim_s=1024,
        latent_dim_l=1024,
        attn_heads_s=16,
        attn_heads_l=16,
        cross_head_s=8,
        cross_head_l=8,
        dim_head_s=64,
        dim_head_l=64,
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
            latent_dim_s,
            depth_s,
            attn_heads_s,
            dim_head_s,
            mlp_dim_s,
            p_dropout_s,
        )
        self.l = VanillaEncoder(
            latent_dim_l,
            depth_l,
            attn_heads_l,
            dim_head_l,
            mlp_dim_l,
            p_dropout_l,
        )
        self.attend_s = CrossAttention(
            latent_dim_s, latent_dim_l, cross_head_s, cross_dim_head_s
        )
        self.attend_l = CrossAttention(
            latent_dim_l, latent_dim_s, cross_head_l, cross_dim_head_l
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
