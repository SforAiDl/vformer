import torch
import torch.nn as nn
from einops import repeat

from vformer.attention import CrossAttention
from vformer.common import BaseClassificationModel
from vformer.encoder.embedding import LinearEmbedding
from vformer.encoder.vanilla import VanillaEncoder


class CrossEncoder(nn.Module):
    def __init__(
        self,
        latent_dim_s=1024,
        latent_dim_l=1024,
        dim_head_s=64,
        dim_head_l=64,
        cross_dim_head_s=64,
        cross_dim_head_l=64,
        depth_s=6,
        depth_l=6,
        attn_heads_s=16,
        attn_heads_l=16,
        cross_head_s=8,
        cross_head_l=8,
        encoder_mlp_dim_s=2048,
        encoder_mlp_dim_l=2048,
        p_dropout_encoder_s=0.0,
        p_dropout_encoder_l=0.0,
    ):
        super().__init__()
        self.s = VanillaEncoder(
            latent_dim_s,
            depth_s,
            attn_heads_s,
            dim_head_s,
            encoder_mlp_dim_s,
            p_dropout_encoder_s,
        )
        self.l = VanillaEncoder(
            latent_dim_l,
            depth_l,
            attn_heads_l,
            dim_head_l,
            encoder_mlp_dim_l,
            p_dropout_encoder_l,
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
