import torch

from .vanilla import VanillaSelfAttention

test_tensor1 = torch.randn(2, 65, 1024)


class GatedPositionalSelfAttention(VanillaSelfAttention):
    def __init__(self, dim, num_heads=8, head_dim=64, p_dropout=0):
        super().__init__(dim, num_heads, head_dim, p_dropout)
