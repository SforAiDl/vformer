import torch

from vformer.models import VanillaViT
from vformer.viz import VITAttentionGradRollout, VITAttentionRollout


def test_attention_rollout():
    img = torch.randn(1, 3, 224, 224)
    layer = "attend"
    model = VanillaViT(img_size=256, patch_size=32, n_classes=10, in_channels=3)
    model_attention_rollout_mean = VITAttentionRollout(model, layer)
    model_attention_rollout_max = VITAttentionRollout(model, layer, "max")
    model_attention_rollout_min = VITAttentionRollout(model, layer, "min")
    _ = model(img)
    _ = model_attention_rollout_mean(img)
    _ = model_attention_rollout_max(img)
    _ = model_attention_rollout_min(img)


def test_attention_grad_rollout():
    img = torch.randn(1, 3, 224, 224)
    layer = "attend"
    model = VanillaViT(img_size=256, patch_size=32, n_classes=10, in_channels=3)
    model_attention_grad_rollout = VITAttentionGradRollout(model, layer)
    _ = model_attention_grad_rollout(img, 0)
