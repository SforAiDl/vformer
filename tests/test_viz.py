import torch
import vformer.viz as viz
from vformer.models import VanillaViT
img = torch.randn(1,3, 224, 224)
layer = "attend"
model = VanillaViT(img_size=256, patch_size=32, n_classes=10, in_channels=3)
model_attention_grad_rollout = viz.VITAttentionGradRollout(model,layer)
model_attention_rollout = viz.VITAttentionRollout(model,layer)
_ = model(img)
_ = model_attention_rollout(img)
_ = model_attention_grad_rollout(img,0)

