import torch

from vformer.attention.vanilla import VanillaSelfAttention
from vformer.attention.window import WindowAttention

test_tensor1 = torch.randn(2, 65, 1024)
test_tensor2 = torch.randn(2, 257, 1024)
test_tensor3 = torch.randn(256, 49, 96)
test_tensor4 = torch.randn(32, 64, 96)


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
