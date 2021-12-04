import torch.nn as nn


class PreNorm(nn.Module):
    """
    Parameters
    ----------
    dim: int
        Dimension of the embedding
    fn:nn.Module
        Attention class
    """

    def __init__(self, dim, fn):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
