import math
import torch
import torch.nn as nn

from ...common import BaseClassificationModel
from ...decoder import MLPDecoder
from ...encoder import MultiScaleBlock, PatchEmbed
from ...utils import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class MultiScaleViT(BaseClassificationModel):
    """
    Implementation of 'Multiscale Vision Transformers'
    https://arxiv.org/abs/2104.11227
    Parameters
    ----------
    """
def __init__(self,
             spatial_size = 224,
             pool_first = False,
             temporal_size = 8,
             in_chans = 3,
             use_2d_patch = False,
             patch_stride = [2,4,4],
             num_classes = 400
             embed_dim = 96
             num_heads = 1
             mlp_ratio = 4.0
             qkv_bias = True
             drop_rate = 0.0
             depth = 16
             drop_path_rate = 0.1
             mode = "conv"
             cls_embed_on = True
             sep_pos_embed = False
             norm_stem = False
             norm_layer = partial(nn.LayerNorm, eps=1e-6)
             patch_kernel = (3, 7, 7)
             patch_stride = (2, 4, 4)
             patch_padding = (1, 3, 3)
             DIM_MUL: [[1, 2.0], [3, 2.0], [14, 2.0]]
             HEAD_MUL: [[1, 2.0], [3, 2.0], [14, 2.0]]
             POOL_KVQ_KERNEL: [3, 3, 3]
             POOL_KV_STRIDE_ADAPTIVE: [1, 8, 8]
             POOL_Q_STRIDE: [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
             ):
        super().__init__()
        self.patch_stride = patch_stride
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        
        self.drop_rate = DROPOUT_RATE
        
        self.cls_embed_on = cls_embed_on
        self.sep_pos_embed = sep_pos_embed
        
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, embed_dim)
                )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(DIM_MUL)):
            dim_mul[DIM_MUL[i][0]] = DIM_MUL[i][1]
        for i in range(len(HEAD_MUL)):
            head_mul[HEAD_MUL[i][0]] = HEAD_MUL[i][1]

        pool_q = [[] for i in range(depth)]
        pool_kv = [[] for i in range(depth)]
        stride_q = [[] for i in range(depth)]
        stride_kv = [[] for i in range(depth)]

        for i in range(len(POOL_Q_STRIDE)):
            stride_q[POOL_Q_STRIDE[i][0]] = POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[POOL_Q_STRIDE[i][0]] = POOL_KVQ_KERNEL
            else:
                pool_q[POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = POOL_KV_STRIDE_ADAPTIVE
            POOL_KV_STRIDE = []
            for i in range(depth):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(POOL_KV_STRIDE)):
            stride_kv[POOL_KV_STRIDE[i][0]] = POOL_KV_STRIDE[
                i
            ][1:]
            if POOL_KVQ_KERNEL is not None:
                pool_kv[
                    POOL_KV_STRIDE[i][0]
                ] = KVQ_KERNEL
            else:
                pool_kv[POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in POOL_KV_STRIDE[i][1:]
                ]

        self.norm_stem = norm_layer(embed_dim) if norm_stem else None

        self.blocks = nn.ModuleList()

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        self.head = head_helper.TransformerBasicHead(
            embed_dim,
            num_classes,
            dropout_rate=self.drop_rate,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,spatial_size,temporal_size, x):
        x = x[0]
        x = self.patch_embed(x)
        
        T = temporal_size // self.patch_stride[0]
        H = spatial_size // self.patch_stride[1]
        W = spatial_size // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)
        if self.cls_embed_on:
            x = x[:, 0]
        else:
            x = x.mean(1)

        x = self.head(x)
        return x
