import torch.nn as nn

from ...utils import pair


class PatchEmbedding(nn.Module):
    """

    Parameters
    ----------
    img_size: int
        Image Size
    patch_size: int
        Patch Size
    in_channels: int
        Number of input channels in the image
    embedding_dim: int
        Number of linear projection output channels
    norm_layer: nn.Module,
        Normalization layer, Default is `nn.LayerNorm`

    """

    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embedding_dim,
        norm_layer=nn.LayerNorm,
    ):
        super(PatchEmbedding, self).__init__()

        self.img_size = pair(img_size)
        self.patch_size = pair(patch_size)
        self.patch_resolution = [
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        ]

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = norm_layer(embedding_dim)

    def forward(self, x):
        """

        Parameters
        ----------
        x:torch.Tensor
            Input tensor

        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying convolution operation with same `kernel_size` and `stride` on input tensor.

        """
        B, C, H, W = x.shape

        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input Image Size {H}*{W} doesnt match model {self.img_size[0]}*{self.img_size[1]}"

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x
