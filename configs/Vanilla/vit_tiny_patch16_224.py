import torch.nn as nn

from vformer.config import LazyCall as L
from vformer.models import VanillaViT

model = L(VanillaViT)(
    img_size=224,
    in_channels=3,
    patch_size=16,
    embedding_dim=192,
    head_dim=192,
    depth=12,
    num_heads=3,
    encoder_mlp_dim=192,
    decoder_config=None,
    pool="cls",
    p_dropout_encoder=0.1,
    p_dropout_embedding=0.1,
    n_classes=1000,
)
