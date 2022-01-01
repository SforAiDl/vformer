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
        self.attn = VanillaSelfAttention()
        self.norm1 = nn.BatchNorm2d(in_dim)
        self.norm2 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
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
    def __init__(
        self,
        patch_size,
        in_channels,
        initial_channels,
    ):
        super().__init__()
        self.stem = nn.Conv2d(
            in_channels, initial_channels, kernel_size=7, padding=3, stride=2
        )
