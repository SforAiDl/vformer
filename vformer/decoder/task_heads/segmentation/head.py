import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

from ....utils import DECODER_REGISTRY


class DoubleConv(nn.Module):
    """
    Module consisting of two convolution layers and activations
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


@DECODER_REGISTRY.register()
class SegmentationHead(nn.Module):
    """
    U-net like up-sampling block
    """

    def __init__(
        self,
        out_channels=1,
        embed_dims=[64, 128, 256, 512],
    ):
        super(SegmentationHead, self).__init__()

        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in reversed(embed_dims):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(embed_dims[-1], embed_dims[-1] * 2)
        self.conv1 = nn.Conv2d(embed_dims[0], out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=4
        )

    def forward(self, skip_connections):

        x = self.bottleneck(skip_connections[-1])
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):

            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.conv1(x)

        return self.conv2(x)
