import torch

from vformer.encoder import (
    CrossEncoder,
    PVTEncoder,
    SwinEncoder,
    SwinEncoderBlock,
    VanillaEncoder,
)
from vformer.functional import PatchMerging

test_tensor1 = torch.randn(2, 65, 1024)
test_tensor2 = torch.randn(3, 3136, 96)
test_tensor3 = torch.randn(3, 5, 128)
test_tensor4 = torch.randn(3, 5, 256)


def test_VanillaEncoder():
    encoder = VanillaEncoder(
        latent_dim=1024, depth=6, heads=16, dim_head=64, mlp_dim=2048
    )
    out = encoder(test_tensor1)
    assert out.shape == test_tensor1.shape  # shape remains same
    del encoder


def test_SwinEncoder():
    # when downsampled
    encoder = SwinEncoder(
        dim=96,
        input_resolution=(224 // 4, 224 // 4),
        depth=2,
        num_heads=3,
        window_size=7,
        downsample=PatchMerging,
    )
    out = encoder(test_tensor2)

    assert out.shape == (3, 784, 192)
    del encoder
    # when not downsampled
    encoder = SwinEncoder(
        dim=96,
        input_resolution=(224 // 4, 224 // 4),
        depth=2,
        num_heads=3,
        window_size=7,
        downsample=None,
        use_checkpoint=True,
    )
    out = encoder(test_tensor2)
    assert out.shape == (3, 3136, 96)
    del encoder


def test_SwinEncoderBlock():
    encoder = SwinEncoderBlock(
        dim=96, input_resolution=(224 // 4, 224 // 4), num_heads=3, window_size=7
    )
    out = encoder(test_tensor2)
    assert out.shape == test_tensor2.shape


def test_CrossEncoder():
    encoder = CrossEncoder(128, 256)
    out = encoder(test_tensor3, test_tensor4)
    assert out[0].shape == test_tensor3.shape
    assert out[1].shape == test_tensor4.shape  # shape remains same
    del encoder


def test_PVTEncoder():
    test_tensor = torch.randn(4, 3136, 64)
    encoder = PVTEncoder(
        dim=64,
        depth=3,
        qkv_bias=True,
        qk_scale=0.0,
        p_dropout=0.0,
        attn_drop=0.1,
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
