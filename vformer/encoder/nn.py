import torch.nn as nn


class FeedForward(nn.Module):
    """
    class FeefForward:
    Inputs
    -----------
    dim- number of dimention in input tensor
    hidden_dim- dimention of hidden linear layer
    p_dropout- probability for dropout layer
    """
    def __init__(self, dim:int, hidden_dim:int, p_dropout:float=0.0):
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
