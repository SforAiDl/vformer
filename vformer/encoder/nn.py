import torch.nn as nn


class FeedForward(nn.Module):
    """

    Parameters
    ----------
    dim: int
        Dimension of the input tensor
    hidden_dim: int, optional
        Dimension of hidden layer
    out_dim: int, optional
        Dimension of the output tensor
    p_dropout: float
        Dropout probability, default=0.0

    """

    def __init__(self, dim, hidden_dim=None, out_dim=None, p_dropout=0.0):
        super().__init__()

        out_dim = out_dim if out_dim is not None else dim
        hidden_dim = hidden_dim if hidden_dim is not None else dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(p_dropout),
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
            Returns output tensor by performing linear operations and activation on input tensor

        """

        return self.net(x)
