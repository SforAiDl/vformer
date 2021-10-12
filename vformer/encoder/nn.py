import torch.nn as nn


class FeedForward(nn.Module):
    """
    Parameters:
    -----------
    dim: int
        Dimension of the input tensor
    hidden_dim: int
        Dimension of hidden layer
    p_dropout: float
        Dropout probability
    """

    def __init__(self, dim, hidden_dim, p_dropout=0.0,fn=nn.GELU):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            fn(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        return self.net(x)
