import torch.nn as nn


class DWConv(nn.Module):
    """
    Depth Wise Convolution

    Parameters
    ----------
    dim: int
        Dimension of the input tensor
    kernel_size_dwconv: int,optional
        Size of the convolution kernel, default is 3
    stride_dwconv: int
        Stride of the convolution, default is 1
    padding_dwconv: int or tuple or str
        Padding added to all sides of the input, default is 1
    bias_dwconv:bool
        Whether to add learnable bias to the output,default is True.

    """

    def __init__(
        self,
        dim,
        kernel_size_dwconv=3,
        stride_dwconv=1,
        padding_dwconv=1,
        bias_dwconv=True,
    ):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size_dwconv,
            stride=stride_dwconv,
            padding=padding_dwconv,
            bias=bias_dwconv,
            groups=dim,
        )

    def forward(self, x, H, W):
        """

        Parameters:
        ----------
        x: torch.Tensor
            Input tensor
        H: int
            Height of image patch
        W: int
            Width of image patch

        Returns:
        ----------
        torch.Tensor
            Returns output tensor after performing depth-wise convolution operation

        """

        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)

        x = x.flatten(2).transpose(1, 2)
        return x
