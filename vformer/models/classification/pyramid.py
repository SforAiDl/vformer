import torch
import torch.nn as nn

from ...decoder import MLPDecoder
from ...encoder import OverlapPatchEmbed, PVTEncoder


class PyramidVisionTransformerV2(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratio=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        decoder_config=None,
        linear=False,
    ):
        super(PyramidVisionTransformerV2, self).__init__()
        self.depths = depths
        assert (
            len(depths) == len(mlp_ratio) == len(num_heads) == len(embed_dims)
        ), "Configurations do not match"
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.patch_embeds = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        self.norms = nn.ModuleList()

        for i in range(len(depths)):
            self.patch_embeds.append(
                OverlapPatchEmbed(
                    img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                )
            )
            self.blocks.append(
                PVTEncoder(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i],
                    linear=linear,
                )
            )
            self.norms.append(norm_layer[embed_dims[i]])

        self.decoder = MLPDecoder(config=decoder_config, n_classes=num_classes)
        if decoder_config is not None:
            assert decoder_config[0] == embed_dims[-1], "Dimensions do not match"

    def forward(self, x):
        B = x.shape[0]

        for i in range(len(self.depths)):
            patch_embed = self.patch_embeds[i]
            block = self.blocks[i]
            norm = self.norms[i]
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i == len(self.depths) - 1:
                x = x.mean(dim=1)
            else:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.decoder(x)
        return x
