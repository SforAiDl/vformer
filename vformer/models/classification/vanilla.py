import torch
import torch.nn as nn
from einops import repeat

from ...common import BaseClassificationModel
from ...decoder import MLPDecoder
from ...encoder import LinearEmbedding, VanillaEncoder


class VanillaViT(BaseClassificationModel):
    """
    Implementation of 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
    https://arxiv.org/abs/2010.11929

    Parameters:
    -----------
    img_size: int
        Size of the image
    patch_size: int
        size of patch
    n_classes: int
        Number of classes in the dataset/ dimension of final layer
    latent_dim: int
    dim_head: int
        Dimension of head
    depth: int
        Number of encoding  blocks
    atten_heads:int
        Number of attention heads in self attention block
    encoder_mlp_dim: int
        Dimension of hidden layer in the feedforward network of encoder
    in_channel: int
        Number of input channels; for rgb images its value is 3; for grey scale its value is 1
    decoder_config:
    pool: str
        A string value which can take values only between {'cls', 'mean'}
    p_dropout_encoder: float
        Probability for dropout layer of encoder
    p_dropout_embedding: float
        Probability for dropout layer of patch embedding
    """

    def __init__(
        self,
        img_size,
        patch_size,
        n_classes,
        latent_dim=1024,
        dim_head=64,
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
        self.pool = lambda x: x.mean(dim=1) if pool == "mean" else x[:, 0]

        if decoder_config is not None:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, decoder_config[0]),
                MLPDecoder(decoder_config, n_classes),
            )
        else:
            self.decoder = MLPDecoder(latent_dim, n_classes)

    def forward(self, x):

        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.embedding_dropout(x)

        x = self.encoder(x)
        x = self.pool(x)
        x = self.decoder(x)

        return x
