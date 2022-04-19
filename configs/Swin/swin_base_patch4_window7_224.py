from vformer.config import LazyCall as L
from vformer.models import SwinTransformer

model = L(SwinTransformer)(
    img_size=224,
    in_channels=3,
    patch_size=4,
    n_classes=1000,
    embedding_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
    window_size=7,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    p_dropout=0.1,
    attn_dropout=0.1,
    drop_path_rate=0.1,
    ape=True,
    decoder_config=None,
    patch_norm=True,
)
