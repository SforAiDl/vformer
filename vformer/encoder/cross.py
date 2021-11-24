import torch
import torch.nn as nn
from einops import repeat

from vformer.attention import CrossAttention
from vformer.common import BaseClassificationModel
from vformer.encoder.embedding import LinearEmbedding
from vformer.encoder.vanilla import VanillaEncoder


class _cross_p(BaseClassificationModel):
    def __init__(
        self,
        img_size,
        patch_size,
        latent_dim=1024,
        dim_head=64,
        depth=6,
        attn_heads=16,
        encoder_mlp_dim=2048,
        in_channels=3,
        p_dropout_encoder=0.0,
        p_dropout_embedding=0.0,
    ):
        super().__init__(img_size, patch_size, in_channels)

        self.patch_embedding = LinearEmbedding(
            latent_dim, self.patch_height, self.patch_width, self.patch_dim
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, latent_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.embedding_dropout = nn.Dropout(p_dropout_embedding)
        self.encoder = VanillaEncoder(
            latent_dim, depth, attn_heads, dim_head, encoder_mlp_dim, p_dropout_encoder
        )

    def forward(self, x):

        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.embedding_dropout(x)
        return x


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
