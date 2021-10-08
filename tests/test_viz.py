import torch
import vformer.viz as viz
from vformer.models import VanillaViT
img = torch.randn(1,3, 224, 224)
layer = "attend"
model = VanillaViT(img_size=256, patch_size=32, n_classes=10, in_channels=3)
model_attention_grad_rollout = viz.VITAttentionGradRollout(model,layer)
model_attention_rollout_mean = viz.VITAttentionRollout(model,layer)
model_attention_rollout_max = viz.VITAttentionRollout(model,layer,"max")
model_attention_rollout_min = viz.VITAttentionRollout(model,layer,"min")
_ = model(img)
_ = model_attention_rollout_mean(img)
_ = model_attention_rollout_max(img)
_ = model_attention_rollout_min(img)
_ = model_attention_grad_rollout(img,0)

