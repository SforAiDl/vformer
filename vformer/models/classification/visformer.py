import einops
import torch
import torch.nn as nn

from vformer.attention import VanillaSelfAttention

from ...utils import MODEL_REGISTRY


# need to add dropout,scale,head,visformerV2_ti
# editted number of heads, head dim_s   need to change
# run one epoch to check if the model is working
class Conv_Block(nn.Module):
    def __init__(self, in_channels, group=8, activation=nn.GELU, drop=0.0):
        super(Conv_Block, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=3,
            padding=1,
            groups=group,
            bias=False,
        )
        self.act2 = activation()
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        xt = x
        xt = self.norm1(xt)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = xt + self.drop(x)
        del xt
        return x


class Attention_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.attn = VanillaSelfAttention(channels, num_heads=6, head_dim=channels // 6)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        xt = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.norm1(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.attn(x)
        x = xt + x
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        xt = x
        x = self.norm2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = xt + x
        del xt
        return x


@MODEL_REGISTRY.register()
class Visformer(nn.Module):
    def __init__(
        self, image_size, n_classes, depth: list, config: str, channel_config: list
    ):
        super().__init__()
        q = 0
        assert (
            len(channel_config) == len(depth) - depth.count(0) + 2
        ), "channel config is not correct"
        self.linear = nn.Linear(channel_config[-1], n_classes)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        image_size = list(image_size)
        self.stem = nn.ModuleList(
            [
                nn.Conv2d(
                    channel_config[q],
                    channel_config[q + 1],
                    kernel_size=7,
                    padding=3,
                    stride=2,
                    bias=False,
                ),
                nn.BatchNorm2d(channel_config[q + 1]),
                nn.ReLU(inplace=True),
            ]
        )
        q += 1
        emb = 2
        image_size = [i // 2 for i in image_size]
        for i in range(len(depth)):
            if depth[i] == 0:
                emb *= 2
                config = "0" + config
                continue
            self.stem.extend(
                [
                    nn.Conv2d(
                        channel_config[q],
                        channel_config[q + 1],
                        kernel_size=emb,
                        stride=emb,
                    ),
                    nn.BatchNorm2d(channel_config[q + 1]),
                ]
            )
            emb = 2
            q += 1
            image_size = [k // depth[i] for k in image_size]
            if config[i] == "0":
                self.stem.extend(
                    [Conv_Block(channel_config[q]) for j in range(depth[i])]
                )
            elif config[i] == "1":
                self.stem.extend(
                    [Attention_Block(channel_config[q]) for j in range(depth[i])]
                )
        self.stem.extend([nn.BatchNorm2d(channel_config[-1]), nn.AdaptiveAvgPool2d(1)])

    def forward(self, x):
        for i in self.stem:
            x = i(x)
        x.squeeze_(2).squeeze_(2)
        x = self.linear(x)
        return x


@MODEL_REGISTRY.register()
def Visformer_S(img_size, n_class):
    return Visformer(img_size, n_class, [0, 7, 4, 4], "011", [3, 32, 192, 384, 768])


@MODEL_REGISTRY.register()
def VisformerV2_S(img_size, n_class):
    return Visformer(
        img_size,
        n_class,
        [1, 10, 14, 3],
        "0011",
        [3, 32, 64, 128, 256, 512],
    )


@MODEL_REGISTRY.register()
def Visformer_Ti(img_size, n_class):
    return Visformer(img_size, n_class, [0, 7, 4, 4], "011", [3, 16, 96, 192, 384])


@MODEL_REGISTRY.register()
def VisformerV2_Ti(img_size, n_class):
    return Visformer(
        img_size,
        n_class,
        [1, 4, 6, 2],
        "0011",
        [3, 24, 48, 96, 192, 384],
    )


print(Visformer_S(224, 1000))
