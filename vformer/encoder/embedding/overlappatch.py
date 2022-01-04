import torch.nn as nn

from ...utils import pair


class OverlapPatchEmbed(nn.Module):
    """

    Parameters
    ----------
    img_size: int
        Image Size
    patch_size: int or tuple(int)
        Patch Size
    stride: int
        Stride of the convolution, default is 4
    in_channels: int
        Number of input channels in the image, default is 3
    embedding_dim: int
        Number of linear projection output channels,default is 768
    norm_layer: nn.Module, optional
        Normalization layer, default is nn.LayerNorm

    """

    def __init__(
        self,
        img_size,
        patch_size,
        stride=4,
        in_channels=3,
        embedding_dim=768,
        norm_layer=nn.LayerNorm,
    ):
        super(OverlapPatchEmbed, self).__init__()

        img_size = pair(img_size)
        patch_size = pair(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.H, self.W = img_size[0] // stride, img_size[1] // stride

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = norm_layer(embedding_dim)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
              Input tensor

        Returns
        ----------
        x: torch.Tensor
            Input tensor
        H: int
            Height of Patch
        W: int
            Width of Patch

        """

        x = self.proj(x)
        H, W = x.shape[2:]
        x = self.norm(x.flatten(2).transpose(1, 2))

        return x, H, W
