import torch

from vformer.decoder import MLPDecoder

test_tensor = torch.randn(2, 3, 100)


def test_MLPDecoder():
    decoder = MLPDecoder(config=100, n_classes=10)
    out = decoder(test_tensor)
    assert out.shape == (2, 3, 10)
    del decoder
    decoder = MLPDecoder(config=(100, 50), n_classes=10)
    out = decoder(test_tensor)
    assert out.shape == (2, 3, 10)
    del decoder
    decoder = MLPDecoder(config=[100, 10], n_classes=5)
    out = decoder(test_tensor)
    assert out.shape == (2, 3, 5)
