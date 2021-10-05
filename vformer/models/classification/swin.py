import torch
import torch.nn as nn

from ...common import BaseClassificationModel
from ...decoder import MLPDecoder
from ...encoder import PatchEmbedding, SwinEncoder
from ...utils import PatchMerging, trunc_normal_


class SwinTransformer(BaseClassificationModel):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        n_classes,
        embed_dim,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=True,
        decoder_config=(1024,),
        patch_norm=True,
    ):
        super(SwinTransformer, self).__init__(
            img_size, patch_size, in_channels, pool="mean"
        )

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.patch_resolution = self.patch_embed.patch_resolution
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.ape = ape

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.encoder = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = SwinEncoder(
                dim=embed_dim * 2 ** i_layer,
                input_resolution=(
                    self.patch_resolution[0] // (2 ** i_layer),
                    self.patch_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < len(depths) - 1 else None,
            )
            self.encoder.append(layer)

        if decoder_config is not None:
            if not isinstance(decoder_config, list):
                decoder_config = list(decoder_config)
            assert (
                decoder_config[0] == embed_dim
            ), "`embed_dim` should be equal to first item of `decoder_config`"
            self.decoder = MLPDecoder(decoder_config, n_classes)
        else:
            self.decoder = MLPDecoder(embed_dim, n_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = norm_layer if norm_layer is not None else nn.Identity

    def forward(self, x):
        x = self.patch_embed(x)

        if self.ape:
            x += self.absolute_pos_embed(x)

        for layer in self.encoder:
            x = layer(x)

        x = self.norm(x)

        x = self.pool(x.transpose(1, 2)).flatten(1)

        x = self.decoder(x)
        return x
