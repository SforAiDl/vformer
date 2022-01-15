import torch
import torch.nn as nn
from ...common import BaseClassificationModel
from einops.layers.torch import Rearrange
from ...encoder.embedding import LinearEmbedding
class ViViT(BaseClassificationModel):
    def __init__(self,img_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4,):
        super(ViViT, self).__init__(img_size=img_size,patch_size=patch_size,num_classes=num_classes,in_channels=in_channels,pool=pool)


        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2



        self.to_patch_embedding = LinearEmbedding()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = (dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
