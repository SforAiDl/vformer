import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from ..utils import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register()
class MemoryEfficientAttention(nn.Module):
    """
    Memory Effecient attention introduced in paper
    `Self-attention Does Not Need O(n2) Memory <https://arxiv.org/abs/2112.05682>`_

    Implementation based on `this repository <https://github.com/AminRezaei0x443/memory-efficient-attention>`_

    Parameters
    -----------
    dim: int
        Dimension of the embedding
    num_heads: int
        Number of the attention heads
    head_dim: int
        Dimension of each head
    p_dropout: float
        Dropout Probability

    """

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=64,
        p_dropout=0.0,
        query_chunk_size=1024,
        key_chunk_size=4096,
    ):
        super().__init__()

        inner_dim = head_dim * num_heads
        project_out = not (num_heads == 1 and head_dim == dim)

        self.num_heads = num_heads
        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(p_dropout))
            if project_out
            else nn.Identity()
        )

    @staticmethod
    def dynamic_slice(x, starts, sizes):
        starts = [
            np.clip(starts[i], 0, x.shape[i] - sizes[i]) for i in range(len(starts))
        ]
        for i, (start, size) in enumerate(zip(starts, sizes)):
            x = torch.index_select(
                x, i, torch.tensor(range(start, start + size), device=x.device)
            )
        return x

    @staticmethod
    def summarize_chunk(query, key, value):
        attn_weights = torch.einsum("...qhd,...khd->...qhk", query, key)
        max_score, _ = torch.max(attn_weights, dim=-1, keepdim=True)
        max_score = max_score.detach()
        exp_weights = torch.exp(attn_weights - max_score)
        exp_values = torch.einsum("...vhf,...qhv->...qhf", value, exp_weights)
        max_score = torch.einsum("...qhk->...qh", max_score)
        return exp_values, exp_weights.sum(dim=-1), max_score

    @staticmethod
    def map_pt(f, xs):
        t = [f(x) for x in xs]
        return tuple(map(torch.stack, zip(*t)))

    @staticmethod
    def scan(f, init, xs, length=None):
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, torch.stack(ys)

    def query_chunk_attention(self, query, key, value):
        num_kv, num_heads, k_features = key.shape[-3:]
        v_features = value.shape[-1]
        key_chunk_size = min(self.key_chunk_size, num_kv)
        query = query / (k_features**0.5)

        def chunk_scanner(chunk_idx):
            key_chunk = self.dynamic_slice(
                key,
                tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0),
                tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features),
            )
            value_chunk = self.dynamic_slice(
                key,
                tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0),
                tuple(value.shape[:-3]) + (key_chunk_size, num_heads, v_features),
            )
            return checkpoint(self.summarize_chunk, query, key_chunk, value_chunk)

        chunk_values, chunk_weights, chunk_max = self.map_pt(
            chunk_scanner, xs=torch.arange(0, num_kv, key_chunk_size)
        )

        global_max, _ = torch.max(chunk_max, 0, keepdim=True)
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values *= torch.unsqueeze(max_diffs, -1)
        chunk_weights *= max_diffs

        all_values = chunk_values.sum(dim=0)
        all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
        return all_values / all_weights

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns output tensor by applying self-attention on input tensor

        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.num_heads), qkv
        )

        num_q, num_heads, q_features = q.shape[-3:]

        def inner_chunk_scanner(chunk_idx, _):
            query_chunk = self.dynamic_slice(
                q,
                tuple([0] * (q.ndim - 3)) + (chunk_idx, 0, 0),
                tuple(q.shape[:-3])
                + (min(self.query_chunk_size, num_q), num_heads, q_features),
            )
            return (
                chunk_idx + self.query_chunk_size,
                self.query_chunk_attention(query_chunk, k, v),
            )

        _, res = self.scan(
            inner_chunk_scanner,
            init=0,
            xs=None,
            length=int(np.ceil(num_q / self.query_chunk_size)),
        )

        rl = [res[i] for i in range(res.shape[0])]
        att = torch.cat(rl, dim=-3)

        out = rearrange(att, "b n h d -> b n (h d)")

        return self.to_out(out)
