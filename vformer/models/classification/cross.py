import torch
import torch.nn as nn
from einops import repeat

from ...common import BaseClassificationModel
from ...decoder import MLPDecoder
from ...encoder import CrossEncoder, LinearEmbedding
from ...utils import MODEL_REGISTRY


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
            torch.randn(1, self.num_patches + 1, latent_dim)
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


@MODEL_REGISTRY.register()
class CrossViT(BaseClassificationModel):
    """
    Implementation of 'CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification'
    https://arxiv.org/abs/2103.14899

    Parameters
    ----------
    img_size: int
        Size of the image
    patch_size_s: int
        Size of the smaller patches
    patch_size_l: int
        Size of the larger patches
    n_classes: int
        Number of classes for classification
    cross_dim_head_s: int
        Dimension of the head of the cross-attention for the smaller patches
    cross_dim_head_l: int
        Dimension of the head of the cross-attention for the larger patches
    latent_dim_s: int
        Dimension of the hidden layer for the smaller patches
    latent_dim_l: int
        Dimension of the hidden layer for the larger patches
    head_dim_s: int
        Dimension of the head of the attention for the smaller patches
    head_dim_l: int
        Dimension of the head of the attention for the larger patches
    depth_s: int
        Number of attention layers in encoder for the smaller patches
    depth_l: int
        Number of attention layers in encoder for the larger patches
    attn_heads_s: int
        Number of attention heads for the smaller patches
    attn_heads_l: int
        Number of attention heads for the larger patches
    cross_head_s: int
        Number of CrossAttention heads for the smaller patches
    cross_head_l: int
        Number of CrossAttention heads for the larger patches
    encoder_mlp_dim_s: int
        Dimension of hidden layer in the encoder for the smaller patches
    encoder_mlp_dim_l: int
        Dimension of hidden layer in the encoder for the larger patches
    in_channels: int
        Number of input channels
    decoder_config_s: int or tuple or list, optional
        Configuration of the decoder for the smaller patches
    decoder_config_l: int or tuple or list, optional
        Configuration of the decoder for the larger patches
    pool_s: {"cls","mean"}
        Feature pooling type for the smaller patches
    pool_l: {"cls","mean"}
        Feature pooling type for the larger patches
    p_dropout_encoder_s: float
        Dropout probability in the encoder for the smaller patches
    p_dropout_encoder_l: float
        Dropout probability in the encoder for the larger patches
    p_dropout_embedding_s: float
        Dropout probability in the embedding layer for the smaller patches
    p_dropout_embedding_l: float
        Dropout probability in the embedding layer for the larger patches
    """

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
        head_dim_s=64,
        head_dim_l=64,
        depth_s=6,
        depth_l=6,
        attn_heads_s=16,
        attn_heads_l=16,
        cross_head_s=8,
        cross_head_l=8,
        encoder_mlp_dim_s=2048,
        encoder_mlp_dim_l=2048,
        in_channels=3,
        decoder_config_s=None,
        decoder_config_l=None,
        pool_s="cls",
        pool_l="cls",
        p_dropout_encoder_s=0.0,
        p_dropout_encoder_l=0.0,
        p_dropout_embedding_s=0.0,
        p_dropout_embedding_l=0.0,
    ):
        super().__init__(img_size, patch_size_s, in_channels, pool_s)
        super().__init__(img_size, patch_size_l, in_channels, pool_l)

        self.s = _cross_p(
            img_size, patch_size_s, latent_dim_s, in_channels, p_dropout_embedding_s
        )
        self.l = _cross_p(
            img_size, patch_size_l, latent_dim_l, in_channels, p_dropout_embedding_l
        )
        self.encoder = CrossEncoder(
            latent_dim_s,
            latent_dim_l,
            head_dim_s,
            head_dim_l,
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
                decoder_config_s = list(decoder_config_s)

            assert (
                decoder_config_s[0] == latent_dim_s
            ), "`latent_dim` should be equal to the first item of `decoder_config`"

            self.decoder_s = MLPDecoder(decoder_config_s, n_classes)

        else:
            self.decoder_s = MLPDecoder(latent_dim_s, n_classes)

        if decoder_config_l is not None:

            if not isinstance(decoder_config_l, list):
                decoder_config_l = list(decoder_config_l)

            assert (
                decoder_config_l[0] == latent_dim_l
            ), "`latent_dim` should be equal to the first item of `decoder_config`"

            self.decoder_l = MLPDecoder(decoder_config_l, n_classes)

        else:
            self.decoder_l = MLPDecoder(latent_dim_l, n_classes)

    def forward(self, img):
        """

        Parameters
        ----------
        img: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of size `n_classes`

        """
        emb_s = self.s(img)
        emb_l = self.l(img)
        emb_s, emb_l = self.encoder(emb_s, emb_l)
        cls_s = self.pool_s(emb_s)
        cls_l = self.pool_l(emb_l)
        n_s = self.decoder_s(cls_s)
        n_l = self.decoder_l(cls_l)
        n = n_s + n_l

        return n
