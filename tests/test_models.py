import torch
import torch.nn as nn

from vformer.utils import MODEL_REGISTRY

models = MODEL_REGISTRY.get_list()

img_3channels_256 = torch.randn(2, 3, 256, 256)
img_3channels_224 = torch.randn(4, 3, 224, 224)
img_1channels_224 = torch.randn(2, 1, 224, 224)

from vformer.models.dense.dpt import DPT, DPTDepthModel


def test_VanillaViT():

    model = MODEL_REGISTRY.get("VanillaViT")(
        img_size=256, patch_size=32, n_classes=10, in_channels=3
    )
    out = model(img_3channels_256)
    assert out.shape == (2, 10)
    del model

    model = MODEL_REGISTRY.get("VanillaViT")(
        img_size=256,
        patch_size=32,
        n_classes=10,
        embedding_dim=1024,
        decoder_config=(1024, 512),
    )
    out = model(img_3channels_256)
    assert out.shape == (2, 10)
    del model


def test_SwinTransformer():

    model = MODEL_REGISTRY.get("SwinTransformer")(
        img_size=224,
        patch_size=4,
        in_channels=3,
        n_classes=1000,
        embedding_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        p_dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
    )
    out = model(img_3channels_224)
    assert out.shape == (4, 1000)
    del model

    # tiny_patch4_window7_224
    model = MODEL_REGISTRY.get("SwinTransformer")(
        img_size=224,
        patch_size=4,
        in_channels=3,
        n_classes=10,
        embedding_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        p_dropout=0.2,
    )
    out = model(img_3channels_224)
    assert out.shape == (4, 10)
    del model

    # tiny_c24_patch4_window8_256
    model = MODEL_REGISTRY.get("SwinTransformer")(
        img_size=256,
        patch_size=4,
        in_channels=3,
        n_classes=10,
        embedding_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=8,
        p_dropout=0.2,
    )
    out = model(img_3channels_256)
    assert out.shape == (2, 10)
    del model

    # for greyscale image
    model = MODEL_REGISTRY.get("SwinTransformer")(
        img_size=224,
        patch_size=4,
        in_channels=1,
        n_classes=10,
        embedding_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        p_dropout=0.2,
    )
    out = model(img_1channels_224)
    assert out.shape == (2, 10)
    del model

    # testing for decoder_config parameter
    model = MODEL_REGISTRY.get("SwinTransformer")(
        img_size=224,
        patch_size=4,
        in_channels=3,
        n_classes=10,
        embedding_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        p_dropout=0.2,
        decoder_config=(768, 256, 10, 2),
    )
    out = model(img_3channels_224)
    del model
    assert out.shape == (4, 10)

    # ape=false
    model = MODEL_REGISTRY.get("SwinTransformer")(
        img_size=224,
        patch_size=4,
        in_channels=3,
        n_classes=10,
        embedding_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        p_dropout=0.2,
        decoder_config=(768, 256, 10, 2),
        ape=False,
    )
    out = model(img_3channels_224)
    assert out.shape == (4, 10)
    del model


def test_CrossVit():

    model = MODEL_REGISTRY.get("CrossViT")(256, 16, 64, 10)
    out = model(img_3channels_256)
    assert out.shape == (2, 10)
    del model

    model = MODEL_REGISTRY.get("CrossViT")(
        256,
        16,
        64,
        10,
        decoder_config_s=(1024, 256, 10),
        decoder_config_l=(1024, 256, 10),
    )
    out = model(img_3channels_256)
    assert out.shape == (2, 10)
    del model


def test_pvt():
    # classification
    model = MODEL_REGISTRY.get("PVTClassification")(
        patch_size=[7, 3, 3, 3],
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratio=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        decoder_config=[512, 10],
        num_classes=10,
    )
    out = model(img_3channels_224)
    assert out.shape == (4, 10)
    del model

    model = MODEL_REGISTRY.get("PVTClassification")(
        patch_size=[7, 3, 3, 3],
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratio=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        decoder_config=512,
        num_classes=10,
    )
    out = model(img_3channels_224)
    assert out.shape == (4, 10)
    del model

    model = MODEL_REGISTRY.get("PVTClassificationV2")(linear=False)
    out = model(img_3channels_224)
    assert out.shape == (4, 1000)
    del model

    model = MODEL_REGISTRY.get("PVTClassificationV2")(num_classes=10)
    out = model(img_3channels_224)
    assert out.shape == (4, 10)
    del model

    model = MODEL_REGISTRY.get("PVTClassificationV2")(num_classes=10)
    out = model(img_3channels_224)
    assert out.shape == (4, 10)
    del model

    model = MODEL_REGISTRY.get("PVTClassification")(num_classes=12)
    out = model(img_3channels_224)
    assert out.shape == (4, 12)
    del model

    model = MODEL_REGISTRY.get("PVTClassificationV2")(
        embedding_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratio=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        linear=True,
    )
    out = model(img_3channels_224)
    assert out.shape == (4, 1000)

    # segmentation
    model = MODEL_REGISTRY.get("PVTSegmentation")()
    outs = model(img_3channels_224)
    assert outs.shape == (
        4,
        1,
        224,
        224,
    ), f"expected: {(4,1,224,224)}, got : {outs.shape}"
    del model

    model = MODEL_REGISTRY.get("PVTSegmentation")()
    outs = model(img_3channels_256)
    assert outs.shape == (
        2,
        1,
        256,
        256,
    ), f"expected: {(4,1,256,256)}, got : {outs.shape}"
    del model

    model = MODEL_REGISTRY.get("PVTSegmentation")()
    outs = model(img_3channels_256)
    assert outs.shape == (
        2,
        1,
        256,
        256,
    ), f"expected: {(4,1,256,256)}, got : {outs.shape}"
    del model

    model = MODEL_REGISTRY.get("PVTSegmentationV2")(return_pyramid=False)
    outs = model(img_3channels_224)
    assert outs.shape == (
        4,
        1,
        224,
        224,
    ), f"expected: {(4,1,224,224)}, got : {outs.shape}"
    del model

    model = MODEL_REGISTRY.get("PVTSegmentationV2")(return_pyramid=True)
    out = model(img_3channels_224)

    model = MODEL_REGISTRY.get("PVTSegmentationV2")(return_pyramid=False)
    outs = model(img_3channels_256)
    assert outs.shape == (
        2,
        1,
        256,
        256,
    ), f"expected: {(4,1,256,256)}, got : {outs.shape}"
    del model

    # detection
    model = MODEL_REGISTRY.get("PVTDetection")()
    outs = model(img_3channels_224)
    del model

    model = MODEL_REGISTRY.get("PVTDetectionV2")()
    outs = model(img_3channels_224)
    del model


def test_cvt():

    model = MODEL_REGISTRY.get("CVT")(img_size=256, patch_size=4, in_channels=3)
    out = model(img_3channels_256)
    assert out.shape == (2, 1000)
    del model

    model = MODEL_REGISTRY.get("CVT")(
        img_size=224,
        patch_size=4,
        in_channels=3,
        seq_pool=False,
        embedding_dim=768,
        num_heads=1,
        mlp_ratio=4.0,
        num_classes=10,
        p_dropout=0.5,
        attn_dropout=0.3,
        drop_path=0.2,
        positional_embedding="sine",
        decoder_config=(768, 12024, 512, 256, 128, 64, 32),
    )
    out = model(img_3channels_224)
    assert out.shape == (4, 10)
    del model

    model = MODEL_REGISTRY.get("CVT")(
        img_size=224,
        in_channels=3,
        patch_size=4,
        positional_embedding="none",
        seq_pool=False,
        decoder_config=None,
    )
    f = model(img_3channels_224)
    assert f.shape == (4, 1000)
    del model

    model = MODEL_REGISTRY.get("CVT")(
        img_size=224,
        in_channels=3,
        patch_size=4,
        positional_embedding="none",
        seq_pool=True,
        decoder_config=768,
    )
    f = model(img_3channels_224)
    assert f.shape == (4, 1000)
    del model


def test_cct():

    model = MODEL_REGISTRY.get("CCT")(img_size=256, patch_size=4, in_channels=3)
    out = model(img_3channels_256)
    assert out.shape == (2, 1000)
    del model

    model = MODEL_REGISTRY.get("CCT")(
        img_size=224,
        patch_size=4,
        in_channels=3,
        seq_pool=False,
        embedding_dim=768,
        num_heads=1,
        mlp_ratio=4.0,
        num_classes=10,
        p_dropout=0.5,
        attn_dropout=0.3,
        drop_path=0.2,
        positional_embedding="sine",
        decoder_config=(768, 12024, 512, 256, 128, 64, 32),
    )
    out = model(img_3channels_224)
    assert out.shape == (4, 10)
    del model

    model = MODEL_REGISTRY.get("CCT")(
        img_size=224,
        in_channels=3,
        patch_size=4,
        positional_embedding="none",
        seq_pool=False,
        decoder_config=None,
    )
    f = model(img_3channels_224)
    assert f.shape == (4, 1000)
    del model

    model = MODEL_REGISTRY.get("CCT")(
        img_size=224,
        in_channels=3,
        patch_size=4,
        positional_embedding="none",
        seq_pool=True,
        decoder_config=768,
    )
    f = model(img_3channels_224)
    assert f.shape == (4, 1000)
    del model

