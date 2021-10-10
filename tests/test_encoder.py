import torch

from vformer.encoder import SwinEncoder, SwinEncoderBlock, VanillaEncoder
from vformer.utils import PatchMerging

test_tensor1 = torch.randn(2, 65, 1024)
test_tensor2 = torch.randn(3, 3136, 96)


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
