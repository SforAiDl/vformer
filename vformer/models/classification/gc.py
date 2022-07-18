import torch
import torch.nn as nn

from ...utils import MODEL_REGISTRY
from ...decoder import MLPDecoder
from ...encoder import PatchEmbedding, GCViTLayer
from ...attention import WindowAttention
from ...attention import WindowAttentionGlobal

from timm.models.layers import trunc_normal_, DropPath, to_2tuple

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ReduceSize(nn.Module):
    def __init__(self, dim,
                 norm_layer=nn.LayerNorm,
                 keep_dim=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2*dim
        self.reduction = nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False)
        self.norm2 = norm_layer(dim_out)
        self.norm1 = norm_layer(dim)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = x + self.conv(x)
        x = self.reduction(x).permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x

class FeatExtract(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x

class GCViT(nn.Module):
    def __init__(self,
                 dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 **kwargs):
        super().__init__()

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(in_chans=in_chans, dim=dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            level = GCViTLayer(dim=int(dim * 2 ** i),
                               depth=depths[i],
                               num_heads=num_heads[i],
                               window_size=window_size[i],
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer,
                               downsample=(i < len(depths) - 1),
                               layer_scale=layer_scale,
                               input_resolution=int(2 ** (-2 - i) * resolution))
            self.levels.append(level)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@MODEL_REGISTRY.register()
def gc_vit_xxtiny(pretrained=False, **kwargs):
    model = GCViT(depths=[2, 2, 6, 2],
                  num_heads=[2, 4, 8, 16],
                  window_size=[7, 7, 14, 7],
                  dim=64,
                  mlp_ratio=3,
                  drop_path_rate=0.2,
                  **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@MODEL_REGISTRY.register()
def gc_vit_xtiny(pretrained=False, **kwargs):
    model = GCViT(depths=[3, 4, 6, 5],
                  num_heads=[2, 4, 8, 16],
                  window_size=[7, 7, 14, 7],
                  dim=64,
                  mlp_ratio=3,
                  drop_path_rate=0.2,
                  **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@MODEL_REGISTRY.register()
def gc_vit_tiny(pretrained=False, **kwargs):
    model = GCViT(depths=[3, 4, 19, 5],
                  num_heads=[2, 4, 8, 16],
                  window_size=[7, 7, 14, 7],
                  dim=64,
                  mlp_ratio=3,
                  drop_path_rate=0.2,
                  **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@MODEL_REGISTRY.register()
def gc_vit_small(pretrained=False, **kwargs):
    model = GCViT(depths=[3, 4, 19, 5],
                  num_heads=[3, 6, 12, 24],
                  window_size=[7, 7, 14, 7],
                  dim=96,
                  mlp_ratio=2,
                  drop_path_rate=0.3,
                  layer_scale=1e-5,
                  **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@MODEL_REGISTRY.register()
def gc_vit_base(pretrained=False, **kwargs):
    model = GCViT(depths=[3, 4, 19, 5],
                  num_heads=[4, 8, 16, 32],
                  window_size=[7, 7, 14, 7],
                  dim=128,
                  mlp_ratio=2,
                  drop_path_rate=0.5,
                  layer_scale=1e-5,
                  **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model
