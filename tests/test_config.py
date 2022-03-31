from vformer.config import LazyCall, instantiate
from vformer.models import PVTSegmentation, SwinTransformer, VanillaViT, ViViTModel2


def test_lazy():
    # classification models
    vanilla_config = LazyCall(VanillaViT)(img_size=224, patch_size=7, n_classes=10)
    swin_config = LazyCall(SwinTransformer)(
        img_size=28, patch_size=4, in_chans=1, n_classes=3
    )
    vivit_config = LazyCall(ViViTModel2)(
        num_frames=16,
        img_size=(64, 64),
        patch_t=8,
        patch_h=4,
        patch_w=4,
        num_classes=10,
        embedding_dim=512,
        depth=3,
        num_heads=4,
        head_dim=32,
        p_dropout=0.0,
        in_channels=1,
    )

    # dense models
    pvt_config = LazyCall(PVTSegmentation)()
    pvt_config["img_size"] = 224

    for i in locals():
        instantiate(i)  # should not raise any error
