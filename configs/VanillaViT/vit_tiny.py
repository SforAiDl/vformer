from vformer.config import LazyCall as L
from vformer.models import VanillaViT

model = L(VanillaViT)(
    patch_size=16,
    embedding_dim=192,
    head_dim=192,
    depth=12,
    attn_heads=3,
    encoder_mlp_dim=192,
    decoder_config=None,
    pool="cls",
    p_dropout_encoder=0.2,
    p_dropout_embedding=0.2,
)
