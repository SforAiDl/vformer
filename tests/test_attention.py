import torch

from vformer.attention.vanilla import VanillaSelfAttention
from vformer.attention.window import WindowAttention

test_tensor1 = torch.randn(2, 65, 1024)
test_tensor2 = torch.randn(2, 257, 1024)
test_tensor3 = torch.randn(256, 196, 96)


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
    pass
