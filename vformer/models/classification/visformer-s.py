import einops
import torch
import torch.nn as nn

from vformer.attention import VanillaSelfAttention


# need to add dropout
class Conv_Block(nn.Module):
    def __init__(self, in_channels, group=1, activation=nn.ReLU, drop=0.0):
        super(Conv_Block, self).__init__()
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
        self.attn = VanillaSelfAttention(channels)
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


class Visformer(nn.Module):
    def __init__(self, image_size, depth, config, channel_config):
        super().__init__()
        q = 0
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
                )
            ]
        )
        q += 1
        image_size = image_size // 2
        for i in range(depth):
            if config[i] == 0:
                self.stem.extend(
                    [Conv_Block(channel_config[q]) for j in range(depth[i])]
                )
            elif config[i] == 1:
                self.stem.extend(
                    [Attention_Block(channel_config[q]) for j in range(depth[i])]
                )
            elif config[i] == 2:
                self.stem.extend(
                    [
                        nn.Conv2d(
                            channel_config[q],
                            channel_config[q + 1],
                            kernel_size=depth[i],
                            stride=depth[i],
                        )
                    ]
                )
                q += 1
                image_size = image_size // depth[i]
            assert q < len(channel_config), "channel configuration not complete"

    def forward(self, x):
        for i in self.stem:
            x = i(x)
        return x
