import tools.visualize as v
from vformer.models import VanillaViT


def test_open_image():
    v.open_image("tests/test_image.jpg", [256, 256])


def test_model_rollout():
    img = v.open_image("tests/test_image.jpg")
    model = VanillaViT(img_size=256, patch_size=32, n_classes=10, in_channels=3)
    v.model_rollout(img, model)
    v.model_rollout(img, model, head_fusion="max")
    v.model_rollout(img, model, head_fusion="min")
    v.model_rollout(img, model, grad_rollout=True)


def test_mask_over_image():
    img = v.open_image("tests/test_image.jpg")
    model = VanillaViT(img_size=256, patch_size=32, n_classes=10, in_channels=3)
    heatmap = v.model_rollout(img, model)
    v.mask_over_image(img, heatmap)
