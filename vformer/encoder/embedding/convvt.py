import torch
import torch.nn as nn
from einops import rearrange


class ConvEmbedding(nn.Module):
    """
    This class converts images to tensors.

    Parameters
    ----------
    patch_size: int, default is 7
        Size of a patch
    in_channels: int, default is 3
        Number of input channels
    embedding_dim: int, default is 64
        Dimension of hidden layer
    stride: int or tuple, default is 4
        Stride of the convolution operation
    padding: int, default is 2
        Padding to all sides of the input
    """

    def __init__(
        self, patch_size=7, in_channels=3, embedding_dim=64, stride=4, padding=2
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)

        self.proj = nn.Conv2d(
            in_channels,
            embedding_dim,
            kernel_size=self.patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.tensor
            Input tensor

        Returns
        -----------
        torch.Tensor
            Returns output tensor (embedding) by applying a convolution operations on input tensor
        """
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x
