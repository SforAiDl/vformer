import torch.nn as nn

from ..functional import DWConv


class PVTFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=None,
        out_dim=None,
        act_layer=nn.GELU,
        drop=0.0,
        linear=False,
        fn=DWConv,
        **kwargs
    ):
        super(PVTFeedForward, self).__init__()
        out_dim = out_dim if out_dim is not None else dim
        hidden_dim = hidden_dim if hidden_dim is not None else dim

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True) if linear else nn.Identity()
        self.fn = fn(dim=hidden_dim, **kwargs)
        self.to_out = nn.Sequential(
            act_layer(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(p=drop),
        )

    def forward(self, x, **kwargs):
        x = self.relu(self.fc1(x))
        x = self.fn(x, **kwargs)
        return self.to_out(x)
