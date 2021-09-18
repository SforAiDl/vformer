import torch
import torch.nn as nn
from einops import repeat

from ...common import BaseClassificationModel
from ...decoder import MLPDecoder
from ...encoder import LinearEmbedding, VanillaEncoder


class VanillaViT(BaseClassificationModel):
    """
    class VanillaViT:
    Inputs:
    ---------------------------
    img_size:image size
    patch_size:size of patch
    n_classes: number of classes in the dataset/ dimension of final layer
    latent_dim:
    dim_head: dimension of head
    depth: number of encoding  blocks
    atten_heads: number of attention heads in self attention block
    encoder_mlp_dim: dimension of hidden layer in the feedforward network of encoder
    in_channel: number of input channels; for rgb images its value is 3; for grey scale its value is 1
    decoder_config:
    pool: a string value which can take values only between {'cls', 'mean'}
    p_dropout_encoder: probability for dropout layer of encoder
    p_dropout_embedding: probability for dropout layer of patch embedding
    """
    def __init__(
        self,
        img_size:int,
        patch_size:int,
        n_classes:int,
        latent_dim:int=1024,
        dim_head:int=64,
        depth:int=6,
        attn_heads:int=16,
        encoder_mlp_dim:int=2048,
        in_channels:int=3,
        decoder_config=None,
        pool:str="cls",
        p_dropout_encoder:float=0.0,
        p_dropout_embedding:float=0.0,
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
            self.decoder = nn.Sequential(nn.Linear(latent_dim,decoder_config[0]),MLPDecoder(decoder_config, n_classes))
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
