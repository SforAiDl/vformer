import torch
import torch.nn as nn


class CVTEmbedding(nn.Module):
    """
    Embedding / Tokenizer for Compact Vision Transformer
    """

    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        in_chans=3,
        out_chans=64,
        in_planes=64,
        activation=None,
        max_pool=True,
        conv_bias=False,
    ):
        super(CVTEmbedding, self).__init__()

        n_filter_list = (
            [in_chans] + [in_planes for _ in range(n_conv_layers - 1)] + [out_chans]
        )
        self.conv_layers = nn.ModuleList([])
        for i in range(n_conv_layers):
            self.conv_layers.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            n_filter_list[i],
                            n_filter_list[i + 1],
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=conv_bias,
                        ),
                        nn.Identity() if activation is None else activation(),
                        nn.MaxPool2d(
                            kernel_size=pooling_kernel_size,
                            stride=pooling_stride,
                            padding=pooling_padding,
                        )
                        if max_pool
                        else nn.Identity(),
                    ]
                )
            )

        self.flatten = nn.Flatten(2, 3)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        for conv2d, activation, maxpool in self.conv_layers:

            x = maxpool(activation(conv2d(x)))

            return self.flatten(x).transpose(-2, -1)
