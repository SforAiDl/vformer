import torch

from .utils import pair


def cyclicshift(input, shift_size, dims=None):
    """
    Parameters
    ----------
    input: torch.Tensor
        input tensor
    shift_size: int or tuple(int)
        Number of places by which input tensor is shifted
    dims: int or tuple(int),optional
        Axis along which to roll
    """

    return torch.roll(
        input, shifts=pair(shift_size), dims=(1, 2) if dims == None else dims
    )


def window_partition(x, window_size):
    """
    Parameters
    ----------
    x: torch.Tensor
        input tensor
    window_size: int
        window size
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Parameters
    ----------
    windows: torch.Tensor
    window_size: int
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_position_bias_index(window_size):
    """
    Parameters
    ----------
    window_size: int or tuple[int]
        Window size
    """
    window_size = pair(window_size)
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = (
        coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)
    return relative_position_index


def create_mask(window_size, shift_size, H, W):
    """
    Parameters
    ----------
    window_size: int
        Window Size
    shift_size: int
        Shift_size

    """
    img_mask = torch.zeros(1, H, W, 1)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask
