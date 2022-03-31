import torch

from vformer.config import LazyCall, instantiate
from vformer.models import PVTSegmentation, SwinTransformer, VanillaViT, ViViTModel2


def test_lazy():
    # classification models
    vanilla_config = LazyCall(VanillaViT)(img_size=224, patch_size=7, n_classes=10)
    swin_config = LazyCall(SwinTransformer)(
        img_size=224,
        patch_size=4,
        in_channels=3,
        n_classes=10,
        embedding_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        p_dropout=0.2,
    )
    vivit_config = LazyCall(ViViTModel2)(
        img_size=224,
        in_channels=3,
        patch_size=16,
        embedding_dim=192,
        depth=4,
        num_heads=3,
        head_dim=64,
        num_frames=1,
        n_classes=10,
    )

    # dense models
    pvt_config = LazyCall(PVTSegmentation)()
    pvt_config["img_size"] = 224
    rand_img_tensor = torch.randn(4, 3, 224, 224)
    rand_vdo_tensor = torch.randn([32, 16, 3, 224, 224])

    vanilla_vit = instantiate(vanilla_config)
    swin_vit = instantiate(swin_config)
    vivit = instantiate(vivit_config)

    pvt = instantiate(pvt_config)

    assert vanilla_vit(rand_img_tensor).shape == (4, 10)
    assert swin_vit(rand_img_tensor).shape == (4, 10)
    assert pvt(rand_img_tensor).shape == (4, 1, 224, 224)
    assert vivit(rand_vdo_tensor).shape == (32, 10)
