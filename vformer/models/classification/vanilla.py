import torch
import torch.nn as nn
from einops import repeat

from ...common import BaseClassificationModel
from ...decoder import MLPDecoder
from ...encoder import LinearEmbedding, PosEmbedding, VanillaEncoder
from ...utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class VanillaViT(BaseClassificationModel):
    """
    Implementation of 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
    https://arxiv.org/abs/2010.11929

    Parameters
    ----------
    img_size: int
        Size of the image
    patch_size: int
        Size of a patch
    n_classes: int
        Number of classes for classification
    embedding_dim: int
        Dimension of hidden layer
    head_dim: int
        Dimension of the attention head
    depth: int
        Number of attention layers in the encoder
    attn_heads:int
        Number of the attention heads
    encoder_mlp_dim: int
        Dimension of hidden layer in the encoder
    in_channels: int
        Number of input channels
    decoder_config: int or tuple or list, optional
        Configuration of the decoder. If None, the default configuration is used.
    pool: {"cls","mean"}
        Feature pooling type
    p_dropout_encoder: float
        Dropout probability in the encoder
    p_dropout_embedding: float
        Dropout probability in the embedding layer
    """

    def __init__(
        self,
        img_size,
        patch_size,
        n_classes,
        embedding_dim=1024,
        head_dim=64,
        depth=6,
        attn_heads=16,
        encoder_mlp_dim=2048,
        in_channels=3,
        decoder_config=None,
        pool="cls",
        p_dropout_encoder=0.0,
        p_dropout_embedding=0.0,
    ):
        super().__init__(img_size, patch_size, in_channels, pool)

        self.patch_embedding = LinearEmbedding(
            embedding_dim, self.patch_height, self.patch_width, self.patch_dim
        )

        self.pos_embedding = PosEmbedding(
            shape=self.num_patches + 1,
            dim=embedding_dim,
            drop=p_dropout_embedding,
            sinusoidal=False,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.encoder = VanillaEncoder(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=attn_heads,
            head_dim=head_dim,
            mlp_dim=encoder_mlp_dim,
            p_dropout=p_dropout_encoder,
        )
        self.pool = lambda x: x.mean(dim=1) if pool == "mean" else x[:, 0]

        if decoder_config is not None:

            if not isinstance(decoder_config, list):
                decoder_config = list(decoder_config)

            assert (
                decoder_config[0] == embedding_dim
            ), "`embedding_dim` should be equal to the first item of `decoder_config`"

            self.decoder = MLPDecoder(decoder_config, n_classes)

        else:
            self.decoder = MLPDecoder(embedding_dim, n_classes)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of size `n_classes`

        """
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.decoder(x)

        return x
