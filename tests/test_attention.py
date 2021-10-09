import torch

from vformer.attention.vanilla import VanillaSelfAttention
from vformer.attention.window import WindowAttention

test_tensor1 = torch.randn(2, 65, 1024)
test_tensor2 = torch.randn(2, 257, 1024)
test_tensor3 = torch.randn(256, 49, 96)
test_tensor4 = torch.randn(32, 64, 96)


def test_VanillaSelfAttention():
    attention = VanillaSelfAttention(dim=1024)
    _ = attention(test_tensor1)

    assert _.shape == (2, 65, 1024)
    del attention
    attention = VanillaSelfAttention(dim=1024, heads=16)
    _ = attention(test_tensor2)
    assert _.shape == (2, 257, 1024)
    del attention


def test_WindowAttention():
    attention = WindowAttention(dim=96, window_size=7, num_heads=3)
    _ = attention(test_tensor3)
    assert _.shape == test_tensor3.shape
    del attention

    attention = WindowAttention(dim=96, window_size=8, num_heads=4)
    _ = attention(test_tensor4)
    assert _.shape == test_tensor4.shape
    del attention
