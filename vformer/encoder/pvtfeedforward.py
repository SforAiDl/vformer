import torch.nn as nn

from ..functional import DWConv


class PVTFeedForward(nn.Module):
    """
    dim: int
        Dimension of the input tensor
    hidden_dim: int, optional
        Dimension of hidden layer
    out_dim:int, optional
        Dimension of output tensor
    act_layer: Activation class
        Activation Layer, default is nn.GELU
    p_dropout: float
        Dropout probability/rate, default is 0.0
    linear: bool
        default=False
    use_dwconv: bool
        default=False
    """

    def __init__(
        self,
        dim,
        hidden_dim=None,
        out_dim=None,
        act_layer=nn.GELU,
        p_dropout=0.0,
        linear=False,
        use_dwconv=False,
        **kwargs
    ):
        super(PVTFeedForward, self).__init__()
        out_dim = out_dim if out_dim is not None else dim
        hidden_dim = hidden_dim if hidden_dim is not None else dim
        self.use_dwconv = use_dwconv
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True) if linear else nn.Identity()
        if use_dwconv:
            self.dw_conv = DWConv(dim=hidden_dim, **kwargs)
        self.to_out = nn.Sequential(
            act_layer(),
            nn.Dropout(p=p_dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(p=p_dropout),
        )

    def forward(self, x, **kwargs):
        x = self.relu(self.fc1(x))
        if self.use_dwconv:
            x = self.dw_conv(x, **kwargs)
        return self.to_out(x)
