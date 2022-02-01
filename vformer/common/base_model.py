import torch.nn as nn

from ..utils import pair


class BaseClassificationModel(nn.Module):
    """
    img_size: int
        Size of the image
    patch_size: int or tuple(int)
        Size of the patch
    in_channels: int
        Number of channels in input image
    pool: {"mean","cls"}
        Feature pooling type
    """

    def __init__(self, img_size, patch_size, in_channels=3, pool="cls"):
        super(BaseClassificationModel, self).__init__()

        img_height, img_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            img_height % patch_height == 0 and img_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (img_height // patch_height) * (img_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patches = num_patches
        self.patch_dim = patch_dim

        assert pool in {
            "cls",
            "mean",
        }, "Feature pooling type must be either cls (cls token) or mean (mean pooling)"
        self.pool = pool
