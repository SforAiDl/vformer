import torch.nn as nn


class PreNorm(nn.Module):
    """
    Parameters
    ----------
    dim: int
        Dimension of the embedding
    fn:nn.Module
        Attention class
    context_dim: int
        Dimension of the context array used in cross attention
    """

    def __init__(self, dim, fn, context_dim=None):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.context_norm = (
            nn.LayerNorm(context_dim) if context_dim is not None else None
        )
        self.fn = fn

    def forward(self, x, **kwargs):
        if "context" in kwargs.keys() and kwargs["context"] is not None:
            normed_context = self.context_norm(kwargs["context"])
            kwargs.update(context=normed_context)
        return self.fn(self.norm(x), **kwargs)
