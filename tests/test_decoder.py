import torch

from vformer.utils import DECODER_REGISTRY

decoder_modules = DECODER_REGISTRY.get_list()


def test_MLPDecoder():

    test_tensor = torch.randn(2, 3, 100)

    decoder = DECODER_REGISTRY.get("MLPDecoder")(config=100, n_classes=10)
    out = decoder(test_tensor)
    assert out.shape == (2, 3, 10)
    del decoder

    decoder = DECODER_REGISTRY.get("MLPDecoder")(config=(100, 50), n_classes=10)
    out = decoder(test_tensor)
    assert out.shape == (2, 3, 10)
    del decoder

    decoder = DECODER_REGISTRY.get("MLPDecoder")(config=[100, 10], n_classes=5)
    out = decoder(test_tensor)
    assert out.shape == (2, 3, 5)


def test_SegmentationHead():

    test_tensor_segmentation_head_256 = [
        torch.randn([2, 64, 64, 64]),
        torch.randn([2, 128, 32, 32]),
        torch.randn([2, 256, 16, 16]),
        torch.randn([2, 512, 8, 8]),
    ]
    test_tensor_segmentation_head_224 = [
        torch.randn([2, 64, 56, 56]),
        torch.randn([2, 128, 28, 28]),
        torch.randn([2, 256, 14, 14]),
        torch.randn([2, 512, 7, 7]),
    ]
    test_tensor_segmentation_head = [
        torch.randn([3, 128, 96, 96]),
        torch.randn([3, 256, 48, 48]),
        torch.randn([3, 512, 24, 24]),
        torch.randn([3, 1024, 12, 12]),
    ]

    head = DECODER_REGISTRY.get("SegmentationHead")(
        out_channels=1,
    )
    out = head(test_tensor_segmentation_head_256)
    assert out.shape == (2, 1, 256, 256)

    head = DECODER_REGISTRY.get("SegmentationHead")(
        out_channels=10,
    )
    out = head(test_tensor_segmentation_head_224)
    assert out.shape == (2, 10, 224, 224)

    head = DECODER_REGISTRY.get("SegmentationHead")(
        out_channels=2, embed_dims=[128, 256, 512, 1024]
    )
    out = head(test_tensor_segmentation_head)
    assert out.shape == (3, 2, 384, 384)
