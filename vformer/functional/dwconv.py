import torch.nn as nn


class DWConv(nn.Module):
    """
    Depth Wise Convolution
    """

    def __init__(
        self, dim, kernel_size_dwconv=3, stride_dwconv=1, padding_dw=1, bias_dwconv=True
    ):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size_dwconv,
            stride=stride_dwconv,
            padding=padding_dw,
            bias=bias_dwconv,
            groups=dim,
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
