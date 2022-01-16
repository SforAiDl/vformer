import torch

from vformer.utils import ATTENTION_REGISTRY

attention_modules = ATTENTION_REGISTRY.get_list()


def test_VanillaSelfAttention():

    test_tensor1 = torch.randn(2, 65, 1024)
    test_tensor2 = torch.randn(2, 257, 1024)

    attention = ATTENTION_REGISTRY.get("VanillaSelfAttention")(dim=1024)
    out = attention(test_tensor1)
    assert out.shape == (2, 65, 1024)
    del attention

    attention = ATTENTION_REGISTRY.get("VanillaSelfAttention")(dim=1024, num_heads=16)
    out = attention(test_tensor2)
    assert out.shape == (2, 257, 1024)
    del attention


def test_WindowAttention():

    test_tensor1 = torch.randn(256, 49, 96)
    test_tensor2 = torch.randn(32, 64, 96)

    attention = ATTENTION_REGISTRY.get("WindowAttention")(
        dim=96, window_size=7, num_heads=3
    )
    out = attention(test_tensor1)
    assert out.shape == test_tensor1.shape
    del attention

    attention = ATTENTION_REGISTRY.get("WindowAttention")(
        dim=96, window_size=8, num_heads=4
    )
    out = attention(test_tensor2)
    assert out.shape == test_tensor2.shape
    del attention


def test_CrossAttention():

    test_tensor1 = torch.randn(64, 1, 64)
    test_tensor2 = torch.randn(64, 24, 128)

    attention = ATTENTION_REGISTRY.get("CrossAttention")(64, 128, 64)
    out = attention(test_tensor1, test_tensor2)
    assert out.shape == test_tensor1.shape
    del attention

def test_MultiScaleAttention():

    test_tensor1 = torch.randn(96,56,56)
    test_tensor2 = torch.randn(768,14,14)
    thw  = [2,2,2]

    attention = ATTENTION_REGISTRY.get("MultiScaleAttention")(dim=56)
    out,_ = attention(test_tensor1, thw)
    assert out.shape == (96,56,56)
    del attention


def test_SpatialAttention():

    test_tensor1 = torch.randn(4, 3136, 64)
    test_tensor2 = torch.randn(4, 50, 512)

    attention = ATTENTION_REGISTRY.get("SpatialAttention")(
        dim=64,
        num_heads=1,
        sr_ratio=8,
    )
    out = attention(test_tensor1, H=56, W=56)
    assert out.shape == test_tensor1.shape
    del attention

    attention = ATTENTION_REGISTRY.get("SpatialAttention")(
        dim=512, num_heads=8, sr_ratio=1, linear=False
    )
    out = attention(test_tensor2, H=7, W=7)
    assert out.shape == test_tensor2.shape
    del attention

    attention = ATTENTION_REGISTRY.get("SpatialAttention")(
        dim=64, num_heads=1, sr_ratio=8, linear=True
    )
    out = attention(test_tensor1, 56, 56)
    assert out.shape == test_tensor1.shape
