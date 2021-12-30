import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from ..utils import ATTENTION_REGISTRY, get_relative_position_bias_index, pair


@ATTENTION_REGISTRY.register()
class WindowAttention(nn.Module):
    """
    Parameters
    ----------
    dim: int
        Number of input channels.
    window_size : int or tuple[int]
        The height and width of the window.
    num_heads: int
        Number of attention heads.
    qkv_bias :bool, default is True
        If True, add a learnable bias to query, key, value.
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 if set
    attn_dropout: float, optional
        Dropout rate
    proj_dropout: float, optional
        Dropout rate

    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = pair(window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv_bias = True
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads
            )
        )
        relative_position_index = get_relative_position_bias_index(self.window_size)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out_1 = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(attn_dropout))
        self.to_out_2 = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_dropout))
        trunc_normal_(self.relative_position_bias_table, std=0.2)

    def forward(self, x, mask=None):
        """

        Parameters
        ----------
        x: torch.Tensor
            input Tensor
        mask: torch.Tensor
            Attention mask used for shifted window attention, if None, window attention will be used,
            else attention mask will be taken into consideration.
            for better understanding you may refer `this <https://github.com/microsoft/Swin-Transformer/issues/38>`

        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying Window-Attention or Shifted-Window-Attention on input tensor

        """

        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                B_,
                N,
                3,
                self.num_heads,
                C // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = (
            self.relative_position_bias_table[self.relative_position_index.view(-1)]
            .view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            .permute(2, 0, 1)
            .contiguous()
        )
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.to_out_1(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.to_out_2(x)

        return x
