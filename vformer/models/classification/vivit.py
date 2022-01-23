import torch
import torch.nn as nn
from einops import rearrange, repeat

from ...common.base_model import BaseClassificationModel
from ...decoder.mlp import MLPDecoder
from ...encoder.embedding import LinearVideoEmbedding, PosEmbedding, TubeletEmbedding
from ...encoder.vanilla import VanillaEncoder
from ...utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ViViT(BaseClassificationModel):
    def __init__(
        self,
        img_size,
        in_channels,
        patch_size,
        embedding_dim,
        num_frames,
        depth,
        num_heads,
        head_dim,
        n_classes,
        mlp_dim=None,
        pool="cls",
        p_dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.02,
    ):
        super(ViViT, self).__init__(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            pool=pool,
        )

        patch_dim = in_channels * patch_size ** 2
        self.patch_embedding = LinearVideoEmbedding(
            embedding_dim=embedding_dim,
            patch_height=patch_size,
            patch_width=patch_size,
            patch_dim=patch_dim,
        )

        self.pos_embedding = PosEmbedding(
            shape=[num_frames, self.num_patches + 1], dim=embedding_dim, drop=p_dropout
        )

        self.space_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )  # this is similar to using cls token in vanilla vision transformer
        self.spatial_transformer = VanillaEncoder(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            p_dropout=p_dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
        )

        self.time_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.temporal_transformer = VanillaEncoder(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            p_dropout=p_dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
        )

        self.decoder = MLPDecoder(
            config=[
                embedding_dim,
            ],
            n_classes=n_classes,
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        (
            b,
            t,
            n,
            d,
        ) = x.shape  # shape of x will be number of videos,time,num_frames,embedding dim
        cls_space_tokens = repeat(self.space_token, "() n d -> b t n d", b=b, t=t)

        x = nn.Parameter(torch.cat((cls_space_tokens, x), dim=2))
        x = self.pos_embedding(x)

        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.spatial_transformer(x)
        x = rearrange(x[:, 0], "(b t) ... -> b t ...", b=b)

        cls_temporal_tokens = repeat(self.time_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.decoder(x)

        return x
