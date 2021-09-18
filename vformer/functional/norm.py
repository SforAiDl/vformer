import torch.nn as nn


class PreNorm(nn.Module):
    """
    class Prenorm:
    Inputs-
    ---------------
    dim: embeding dimention
    fn: Attention class (should be inherited from nn.module)

    Output-
    --------------
    forward method returns attention of a normalised input vector
    """
    def __init__(self, dim:int, fn):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
