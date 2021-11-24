import torch
import torch.nn as nn
from einops import repeat

from vformer.common import BaseClassificationModel
from vformer.decoder.mlp import MLPDecoder
from vformer.encoder.cross import CrossEncoder
from vformer.encoder.embedding import LinearEmbedding


class _cross_p(BaseClassificationModel):
    def __init__(
        self,
        img_size,
        patch_size,
        latent_dim=1024,
        in_channels=3,
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

    def forward(self, x):

        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.embedding_dropout(x)
        return x


class CrossViT(BaseClassificationModel):
    def __init__(
        self,
        img_size,
        patch_size_s,
        patch_size_l,
        n_classes,
        cross_dim_head_s=64,
        cross_dim_head_l=64,
        latent_dim_s=1024,
        latent_dim_l=1024,
        dim_head_s=64,
        dim_head_l=64,
        depth_s=6,
        depth_l=6,
        attn_heads_s=16,
        attn_heads_l=16,
        cross_head_s=8,
        cross_head_l=8,
        encoder_mlp_dim_s=2048,
        encoder_mlp_dim_l=2048,
        in_channels_s=3,
        in_channels_l=3,
        decoder_config_s=None,
        decoder_config_l=None,
        pool_s="cls",
        pool_l="cls",
        p_dropout_encoder_s=0.0,
        p_dropout_encoder_l=0.0,
        p_dropout_embedding_s=0.0,
        p_dropout_embedding_l=0.0,
    ):
        super().__init__(img_size, patch_size_s, in_channels_s, pool_s)
        super().__init__(img_size, patch_size_l, in_channels_l, pool_l)
        self.s = _cross_p(
            img_size, patch_size_s, latent_dim_s, in_channels_s, p_dropout_embedding_s
        )
        self.l = _cross_p(
            img_size, patch_size_l, latent_dim_l, in_channels_l, p_dropout_embedding_l
        )
        self.encoder = CrossEncoder(
            latent_dim_s,
            latent_dim_l,
            dim_head_s,
            dim_head_l,
            cross_dim_head_s,
            cross_dim_head_l,
            depth_s,
            depth_l,
            attn_heads_s,
            attn_heads_l,
            cross_head_s,
            cross_head_l,
            encoder_mlp_dim_s,
            encoder_mlp_dim_l,
            p_dropout_encoder_s,
            p_dropout_encoder_l,
        )
        self.pool_s = lambda x: x.mean(dim=1) if pool_s == "mean" else x[:, 0]
        self.pool_l = lambda x: x.mean(dim=1) if pool_l == "mean" else x[:, 0]
        if decoder_config_s is not None:
            if not isinstance(decoder_config_s, list):
                decoder_config = list(decoder_config_l)
            assert (
                decoder_config[0] == latent_dim_s
            ), "`latent_dim` should be equal to the first item of `decoder_config`"
            self.decoder_s = MLPDecoder(decoder_config, n_classes)

        else:
            self.decoder_s = MLPDecoder(latent_dim_s, n_classes)

        if decoder_config_l is not None:
            if not isinstance(decoder_config_l, list):
                decoder_config = list(decoder_config_l)
            assert (
                decoder_config[0] == latent_dim_l
            ), "`latent_dim` should be equal to the first item of `decoder_config`"
            self.decoder_l = MLPDecoder(decoder_config, n_classes)

        else:
            self.decoder_l = MLPDecoder(latent_dim_l, n_classes)

    def forward(self, img):
        emb_s = self.s(img)
        emb_l = self.l(img)
        emb_s, emb_l = self.encoder(emb_s, emb_l)
        cls_s = self.pool_s(emb_s)
        cls_l = self.pool_l(emb_l)
        n_s = self.decoder_s(cls_s)
        n_l = self.decoder_l(cls_l)
        n = n_s + n_l
        return n
