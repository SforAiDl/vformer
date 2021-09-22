import torch.nn as nn


class FeedForward(nn.Module):
    """
    Parameters:
    -----------
    dim: int
        Number of dimentions in input tensor

    hidden_dim:int
        Dimention of hidden linear layer in feedforward class

    p_dropout:float
        Probability for dropout layer
    """

    def __init__(self, dim, hidden_dim, p_dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        return self.net(x)
