import einops
import torch
import torch.nn as nn

from vformer.attention import VanillaSelfAttention


# need to add dropout
class Conv_Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, group=1, activation=nn.ReLU, drop=0.0
    ):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size=1, bias=False)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(
            out_channels * 2,
            out_channels * 2,
            kernel_size=3,
            padding=1,
            groups=group,
            bias=False,
        )
        self.act2 = activation()
        self.conv3 = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=1, bias=False
        )
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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_dim * 4, out_dim, kernel_size=1, bias=False)
        self.attn = VanillaSelfAttention(in_dim)
        self.norm1 = nn.BatchNorm2d(in_dim)
        self.norm2 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b c (h w)")
        xt = x
        x = self.norm1(x)
        x = self.attn(x)
        x = xt + x
        xt = x
        x = self.norm2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = xt + x
        del xt
        return x


class Visformer(nn.Module):
    def __init__(self, in_channels, initial_channels, depth, config):
        super().__init__()
        self.stem = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels, initial_channels, kernel_size=7, padding=3, stride=2
                )
            ]
        )
        self.st1 = nn.ModuleList([Conv_Block(192, 192) for i in range(depth[0])])
        self.st2 = nn.ModuleList([Attention_Block(384, 384) for i in range(depth[1])])
        self.st3 = nn.ModuleList([Attention_Block(768, 768) for i in range(depth[2])])
        for i in range(depth):
            if config[i] == 0:
                self.stem.extend([[Conv_Block(192, 192) for j in range(depth[i])]])
            else:
                self.stem.extend([Attention_Block(384, 384) for j in range(depth[i])])
