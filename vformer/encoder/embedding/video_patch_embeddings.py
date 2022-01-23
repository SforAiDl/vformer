# Video Patch Embedding
# data Dim is in following format - Batch,Time,Channels,Height,Width

# As discussed in paper; this embedding is just Like Vanilla Embedding where we take non overlapping image patches and map them into multidimension embedding

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class LinearVideoEmbedding(nn.Module):
    """

    Parameters
    ----------
    embedding_dim: int
        Dimension of the resultant embedding
    patch_height: int
        Height of the patch
    patch_width: int
        Width of the patch

    """

    def __init__(
        self,
        embedding_dim,
        patch_height,
        patch_width,
        patch_dim,
    ):

        super().__init__()
        self.patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h ph) (w pw) -> b t (h w) (ph pw c)",
                ph=patch_height,
                pw=patch_width,
            ),
            nn.Linear(patch_dim, embedding_dim),
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        ----------
        torch.Tensor
            Returns patch embeddings of size `embedding_dim`

        """

        return self.patch_embedding(x)


#
class TubeletEmbedding(nn.Module):
    """

    Parameters
    ----------
    embedding_dim: int
        Dimension of the resultant embedding
    tubelet_t: int
        Temporal length of single tube/patch
    tubelet_h: int
        Heigth  of single tube/patch
    tubelet_w: int
        Width of single tube/patch

    """

    def __init__(self, embedding_dim, tubelet_t, tubelet_h, tubelet_w, in_channels):
        super(TubeletEmbedding, self).__init__()
        tubelet_dim = in_channels * tubelet_h * tubelet_w * tubelet_t
        self.tubelet_embedding = nn.Sequential(
            Rearrange(
                "b  (t pt) c (h ph) (w pw) -> b t (h w) (pt ph pw c)",
                pt=tubelet_t,
                ph=tubelet_h,
                pw=tubelet_w,
            ),
            nn.Linear(tubelet_dim, embedding_dim),
        )

    def forward(self, x):
        """

        Parameters
        ----------
        x: Torch.tensor
            Input tensor

        """
        return self.tubelet_embedding(x)
