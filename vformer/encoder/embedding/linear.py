from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class LinearEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        patch_height,
        patch_width,
        patch_dim,
    ):
        super().__init__()

        self.patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, embedding_dim),
        )

    def forward(self, x):

        return self.patch_embedding(x)
