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


def test_GatedPositionalSelfAttention():

    test_tensor1 = torch.randn(2, 64, 1024)
    test_tensor2 = torch.randn(2, 256, 1024)

    attention = ATTENTION_REGISTRY.get("GatedPositionalSelfAttention")(dim=1024)
    out = attention(test_tensor1)
    assert out.shape == (2, 64, 1024)
    del attention

    attention = ATTENTION_REGISTRY.get("GatedPositionalSelfAttention")(
        dim=1024, num_heads=16
    )
    out = attention(test_tensor2)
    assert out.shape == (2, 256, 1024)
    del attention


def test_ConvVTAttention():
    test_tensor1 = torch.randn(16, 196, 384)
    attention = ATTENTION_REGISTRY.get("ConvVTAttention")(384, 128, 4, 14)
    out = attention(test_tensor1)
    assert out.shape == torch.Size([16, 196, 128])
    del attention

    test_tensor1 = torch.randn(16, 196, 384)
    attention = ATTENTION_REGISTRY.get("ConvVTAttention")(384, 128, 4, 14, method="avg")
    out = attention(test_tensor1)
    assert out.shape == torch.Size([16, 196, 128])
    del attention
