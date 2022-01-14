import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    arameters
    ----------
    img_size: int
        Image Size
    
    dim_in: int
        Number of input channels in the image
    dim_out: int
        Number of linear projection output channels
    kernel: int
        kernel Size
    stride: int
        stride Size
    padding: int
        padding Size
    conv_2d : bool
        Use nn.Conv2D if true, nn.conv3D if fals3
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(1, 16, 16),
        stride=(1, 4, 4),
        padding=(1, 7, 7),
        conv_2d=False,
    ):
        super().__init__()
        if conv_2d:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.proj = conv(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)
