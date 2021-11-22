import torch

from vformer.attention.spatial import SpatialAttention
from vformer.attention.vanilla import VanillaSelfAttention
from vformer.attention.window import WindowAttention

test_tensor1 = torch.randn(2, 65, 1024)
test_tensor2 = torch.randn(2, 257, 1024)
test_tensor3 = torch.randn(256, 49, 96)
test_tensor4 = torch.randn(32, 64, 96)
test_tensor5 = torch.randn(4, 3136, 64)
test_tensor6 = torch.randn(4, 50, 512)


def test_VanillaSelfAttention():
    attention = VanillaSelfAttention(dim=1024)
    out = attention(test_tensor1)

    assert out.shape == (2, 65, 1024)
    del attention
    attention = VanillaSelfAttention(dim=1024, heads=16)
    out = attention(test_tensor2)
    assert out.shape == (2, 257, 1024)
    del attention


def test_WindowAttention():
    attention = WindowAttention(dim=96, window_size=7, num_heads=3)
    out = attention(test_tensor3)
    assert out.shape == test_tensor3.shape
    del attention

    attention = WindowAttention(dim=96, window_size=8, num_heads=4)
    out = attention(test_tensor4)
    assert out.shape == test_tensor4.shape
    del attention


def test_SpatialAttention():
    attention = SpatialAttention(
        dim=64,
        num_heads=1,
        sr_ratio=8,
    )
    out = attention(test_tensor5, H=56, W=56)
    assert out.shape == test_tensor5.shape
    del attention
    attention = SpatialAttention(dim=512, num_heads=8, sr_ratio=1, linear=False)
    out = attention(test_tensor6, H=7, W=7)
    assert out.shape == test_tensor6.shape
    del attention

    attention = SpatialAttention(dim=64, num_heads=1, sr_ratio=8, linear=True)
    out = attention(test_tensor5, 56, 56)
    assert out.shape == test_tensor5.shape
