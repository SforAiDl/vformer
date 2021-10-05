import torch.nn as nn

from ...utils import pair


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, norm_layer=None):
        self.img_size = pair(img_size)
        self.patch_size = pair(patch_size)
        self.patch_resolution = [
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        ]

        self.proj_ = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input Image Size {H}*{W} doesnt match model {self.img_size[0]}*{self.img_size[1]}"
        x = self.proj_(x).flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)
        return x
