import torch.nn as nn
from timm.models.layers import DropPath

from ..attention import SpatialAttention
from ..common.blocks import DWConv
from ..functional import PreNorm
from ..utils import ENCODER_REGISTRY


class PVTFeedForward(nn.Module):
    """

    Parameters
    ----------
    dim: int
        Dimension of the input tensor
    hidden_dim: int, optional
        Dimension of hidden layer
    out_dim:int, optional
        Dimension of output tensor
    act_layer: nn.Module
        Activation Layer, default is nn.GELU
    p_dropout: float
        Dropout probability/rate, default is 0.0
    linear: bool
        Whether to use linear Spatial attention,default is False
    use_dwconv: bool
        Whether to use Depth-wise convolutions, default is False

    kernel_size_dwconv: int
        `kernel_size` parameter for 2D convolution used in Depth wise convolution
    stride_dwconv: int
        `stride` parameter for 2D convolution used in Depth wise convolution
    padding_dwconv: int
        `padding` parameter for 2D convolution used in Depth wise convolution
    bias_dwconv:bool
        `bias` parameter for 2D convolution used in Depth wise convolution
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

        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        H: int
            Height of image patch
        W: int
            Width of image patch

        Returns
        --------
        torch.Tensor
            Returns output tensor

        """
        x = self.relu(self.fc1(x))

        if self.use_dwconv:
            x = self.dw_conv(x, **kwargs)

        return self.to_out(x)


@ENCODER_REGISTRY.register()
class PVTEncoder(nn.Module):
    """
    Parameters
    ----------
    dim: int
        Dimension of the input tensor
    num_heads: int
        Number of attention heads
    mlp_ratio:
        Ratio of MLP hidden dimension to embedding dimension
    depth: int
        Number of attention layers in the encoder
    qkv_bias: bool
        Whether to add a bias vector to the q,k, and v matrices
    qk_scale:float, optional
        Override default qk scale of head_dim ** -0.5 in Spatial Attention if set
    p_dropout: float
        Dropout probability
    attn_dropout: float
        Dropout probability
    drop_path: tuple(float)
        List of stochastic drop rate
    act_layer: activation layer
        Activation layer
    use_dwconv:bool
        Whether to use depth-wise convolutions in overlap-patch embedding
    sr_ratio: float
        Spatial Reduction ratio
    linear: bool
        Whether to use linear Spatial attention, default is False
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        depth,
        qkv_bias,
        qk_scale,
        p_dropout,
        attn_dropout,
        drop_path,
        act_layer,
        use_dwconv,
        sr_ratio,
        linear=False,
    ):
        super(PVTEncoder, self).__init__()

        self.encoder = nn.ModuleList([])

        for i in range(depth):
            self.encoder.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim=dim,
                            fn=SpatialAttention(
                                dim=dim,
                                num_heads=num_heads,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                attn_drop=attn_dropout,
                                proj_drop=p_dropout,
                                sr_ratio=sr_ratio,
                                linear=linear,
                            ),
                        ),
                        PreNorm(
                            dim=dim,
                            fn=PVTFeedForward(
                                dim=dim,
                                hidden_dim=int(dim * mlp_ratio),
                                act_layer=act_layer,
                                p_dropout=p_dropout,
                                linear=linear,
                                use_dwconv=use_dwconv,
                            ),
                        ),
                    ]
                )
            )
            self.drop_path = (
                DropPath(drop_prob=drop_path[i])
                if drop_path[i] > 0.0
                else nn.Identity()
            )

    def forward(self, x, **kwargs):

        for prenorm_attn, prenorm_ff in self.encoder:
            x = x + self.drop_path(prenorm_attn(x, **kwargs))
            x = x + self.drop_path(prenorm_ff(x, **kwargs))

        return x
