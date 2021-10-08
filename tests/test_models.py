import torch
import torch.nn as nn

from vformer.models import SwinTransformer, VanillaViT

img_3channels_256 = torch.randn(2, 3, 256, 256)
img_3channels_224 = torch.randn(2, 3, 224, 224)
img_1channels_224 = torch.randn(2, 1, 224, 224)


def test_VanillaViT():

    model = VanillaViT(img_size=256, patch_size=32, n_classes=10, in_channels=3)
    _ = model(img_3channels_256)

    model = VanillaViT(
        img_size=256,
        patch_size=32,
        n_classes=10,
        latent_dim=1024,
        decoder_config=(1024, 512),
    )
    _ = model(img_3channels_256)
    del model


def test_SwinTransformer():
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_channels=3,
        n_classes=1000,
        embed_dim=96,
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
        ape=False,
        patch_norm=True,
    )
    _ = model(img_3channels_224)
    del model
    # tiny_patch4_window7_224
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_channels=3,
        n_classes=10,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_rate=0.2,
    )
    _ = model(img_3channels_224)
    del model
    # tiny_c24_patch4_window8_256
    model = SwinTransformer(
        img_size=256,
        patch_size=4,
        in_channels=3,
        n_classes=10,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=8,
        drop_rate=0.2,
    )
    _ = model(img_3channels_256)
    del model
    # for greyscale image
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_channels=1,
        n_classes=10,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_rate=0.2,
    )
    _ = model(img_1channels_224)
    del model
