import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from ...common import BaseClassificationModel
from ...decoder import MLPDecoder
from ...encoder import PatchEmbedding, SwinEncoder
from ...utils import PatchMerging


class SwinTransformer(BaseClassificationModel):
    """
    Implementation of `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    https://arxiv.org/abs/2103.14030v1

    Parameters:
    -----------
    img_size: int
        Size of an Image
    patch_size: int
        Patch Size
    in_channels:int
        Input channels in image, default=3
    n_classes: int
        Number of classes for classification
    embed_dim: int
        Patch Embedding dimension
    depths: tuple[int]
        Depth in each Transformer layer
    num_heads: tuple[int]
        Number of heads in each transformer layer
    window_size: int
        Window Size
    mlp_ratio : float
        Ratio of mlp heads to embedding dimension
    qkv_bias: bool, default= True
        Adds biasto the qkv if true
    qk_scale:  float, optional
    drop_rate: float
        Dropout rate
    attn_drop_rate: float
        Attension dropout rate
    drop_path_rate: float
        Stochastic depth rate
    norm_layer: nn.Module
    ape: bool
        Adds relative/absolute position embedding if true
    decoder_config: int or tuple[int], optional
    patch_norm: bool, optional
        Adds normalisation layer to PatchEmbedding if true
    """

    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        n_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=True,
        decoder_config=None,
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
            norm_layer=norm_layer if patch_norm else nn.Identity,
        )
        self.patch_resolution = self.patch_embed.patch_resolution
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.ape = ape
        num_features = int(embed_dim * 2 ** (len(depths) - 1))

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.encoder = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = SwinEncoder(
                dim=int(embed_dim * (2 ** i_layer)),
                input_resolution=(
                    (self.patch_resolution[0] // (2 ** i_layer)),
                    self.patch_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_scale=qk_scale,
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
                decoder_config[0] == num_features
            ), f"first item of `decoder_config` should be equal to the `num_features`; num_features=embed_dim * 2** (len(depths)-1) which is = {num_features} "
            self.decoder = MLPDecoder(decoder_config, n_classes)
        else:
            self.decoder = MLPDecoder(num_features, n_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = norm_layer(num_features) if norm_layer is not None else nn.Identity
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x += self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.encoder:
            x = layer(x)

        x = self.norm(x)

        x = self.pool(x.transpose(1, 2)).flatten(1)
        x = self.decoder(x)
        return x
