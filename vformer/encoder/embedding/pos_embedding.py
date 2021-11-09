import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from ...utils import pair


class AbsolutePositionEmbedding(nn.Module):
    """
    Parameters:
    -----------
    pos_shape : int or tuple(int)
        The shape of the absolute position embedding.
    pos_dim : int
        The dimension of the absolute position embedding.
    p_dropout : float, optional
        Probability of an element to be zeroed, default is 0.2
    std: float
        Standard deviation for truncated normal distribution
    """

    def __init__(self, pos_shape, pos_dim, p_dropout=0.0, std=0.02):
        super().__init__()

        pos_shape = pair(pos_shape)
        self.pos_shape = pos_shape
        self.pos_dim = pos_dim

        self.pos_embed = nn.Parameter(
            torch.zeros(1, pos_shape[0] * pos_shape[1], pos_dim)
        )
        self.drop = nn.Dropout(p=p_dropout)
        trunc_normal_(self.pos_embed, std=std)

    def resize_pos_embed(self, pos_embed, shape, mode="bilinear", **kwargs):
        """
        Parameters:
        -----------
            pos_embed : torch.Tensor
                Position embedding weights
            shape : tuple
                Required shape
            mode : str  ('nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear')
                Algorithm used for up/down sampling, default is 'bilinear'
        """
        assert pos_embed.ndim == 3, "shape of pos_embed must be [B, L, C]"
        pos_h, pos_w = self.pos_shape
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
        pos_embed_weight = (
            pos_embed_weight.reshape(1, pos_h, pos_w, self.pos_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pos_embed_weight = F.interpolate(
            pos_embed_weight, size=shape, mode=mode, **kwargs
        )
        pos_embed_weight = (
            torch.flatten(pos_embed_weight, 2).transpose(1, 2).contiguous()
        )
        pos_embed = pos_embed_weight

        return pos_embed

    def forward(self, x, H, W, mode="bilinear"):
        pos_embed = self.resize_pos_embed(self.pos_embed, (H, W), mode)
        return self.drop(x + pos_embed)
