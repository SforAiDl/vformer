import torch
import torch.nn as nn

from vformer.encoder.embedding import TubeletEmbedding
from vformer.functional import PatchMerging
from vformer.utils import ENCODER_REGISTRY

encoder_modules = ENCODER_REGISTRY.get_list()


def test_VanillaEncoder():

    test_tensor = torch.randn(2, 65, 1024)
    encoder = ENCODER_REGISTRY.get("VanillaEncoder")(
        embedding_dim=1024, depth=6, num_heads=16, head_dim=64, mlp_dim=2048
    )
    out = encoder(test_tensor)
    assert out.shape == test_tensor.shape  # shape remains same
    del encoder, test_tensor


def test_SwinEncoder():

    test_tensor = torch.randn(3, 3136, 96)

    # when downsampled
    encoder = ENCODER_REGISTRY.get("SwinEncoder")(
        dim=96,
        input_resolution=(224 // 4, 224 // 4),
        depth=2,
        num_heads=3,
        window_size=7,
        downsample=PatchMerging,
    )
    out = encoder(test_tensor)

    assert out.shape == (3, 784, 192)
    del encoder

    # when not downsampled
    encoder = ENCODER_REGISTRY.get("SwinEncoder")(
        dim=96,
        input_resolution=(224 // 4, 224 // 4),
        depth=2,
        num_heads=3,
        window_size=7,
        downsample=None,
        use_checkpoint=True,
    )
    out = encoder(test_tensor)
    assert out.shape == (3, 3136, 96)
    del encoder

    encoder_block = ENCODER_REGISTRY.get("SwinEncoderBlock")(
        dim=96, input_resolution=(224 // 4, 224 // 4), num_heads=3, window_size=7
    )
    out = encoder_block(test_tensor)
    assert out.shape == test_tensor.shape
    del encoder_block


def test_PVTEncoder():

    test_tensor = torch.randn(4, 3136, 64)

    encoder = ENCODER_REGISTRY.get("PVTEncoder")(
        dim=64,
        depth=3,
        qkv_bias=True,
        qk_scale=0.0,
        p_dropout=0.0,
        attn_dropout=0.1,
        drop_path=[0.0] * 3,
        act_layer=nn.GELU,
        sr_ratio=1,
        linear=False,
        use_dwconv=False,
        num_heads=1,
        mlp_ratio=4,
    )
    out = encoder(test_tensor, H=56, W=56)
    assert out.shape == test_tensor.shape
    del encoder


def test_CrossEncoder():

    test_tensor1 = torch.randn(3, 5, 128)
    test_tensor2 = torch.randn(3, 5, 256)

    encoder = ENCODER_REGISTRY.get("CrossEncoder")(128, 256)
    out = encoder(test_tensor1, test_tensor2)
    assert out[0].shape == test_tensor1.shape
    assert out[1].shape == test_tensor2.shape
    del encoder


def test_ConViTEncoder():

    test_tensor = torch.randn(2, 64, 1024)
    encoder = ENCODER_REGISTRY.get("ConViTEncoder")(
        embedding_dim=1024, depth=6, num_heads=16, head_dim=64, mlp_dim=2048
    )
    out = encoder(test_tensor)
    assert out.shape == test_tensor.shape  # shape remains same
    del encoder, test_tensor


def test_ConvVTStage():
    test_tensor1 = torch.randn(32, 3, 224, 224)

    encoder = ENCODER_REGISTRY.get("ConvVTStage")(
        patch_size=7,
        patch_stride=4,
        patch_padding=2,
        img_size=56,
        embedding_dim=64,
        depth=1,
        num_heads=1,
        with_cls_token=True,
    )
    out, cls_tokens = encoder(test_tensor1)
    assert out.shape == torch.Size([32, 64, 56, 56])
    del encoder

    test_tensor1 = torch.randn(32, 3, 224, 224)

    encoder = ENCODER_REGISTRY.get("ConvVTStage")(
        patch_size=7,
        patch_stride=4,
        patch_padding=2,
        img_size=56,
        embedding_dim=64,
        depth=1,
        num_heads=1,
        with_cls_token=True,
        init="xavier",
    )
    out, cls_tokens = encoder(test_tensor1)
    assert out.shape == torch.Size([32, 64, 56, 56])
    del encoder


def test_TubeletEmbedding():
    test_tensor = torch.randn(
        7, 20, 3, 224, 224
    )  # batch_size,time,in_channels,height,width
    embedding = TubeletEmbedding(
        embedding_dim=192, tubelet_w=16, tubelet_t=5, tubelet_h=16, in_channels=3
    )
    out = embedding(test_tensor)
    assert out.shape == (
        7,
        4,
        196,
        192,
    )  # batch,time/tubelet_t,height*width/(tubelet_h,tubelet_w),embeeding_dim
    del embedding

    test_tensor = torch.randn(11, 15, 1, 28, 28)
    embedding = TubeletEmbedding(96, 5, 7, 7, 1)
    out = embedding(test_tensor)
    assert out.shape == (11, 3, 16, 96)
    del embedding


def test_ViViTEncoder():

    encoder = ENCODER_REGISTRY.get("ViViTEncoder")(
        dim=192, num_heads=3, head_dim=64, p_dropout=0.0, depth=3
    )

    test_tensor = torch.randn(7, 20, 196, 192)
    logits = encoder(test_tensor)
    assert logits.shape == (7, 3920, 192)
