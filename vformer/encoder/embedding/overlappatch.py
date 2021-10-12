import torch.nn as nn
from ...utils import pair

class OverlapPatchEmbed(nn.Module):
    """

    """

    def __init__(self, img_size,
                 patch_size,
                 stride=4,
                 in_channels=3,
                 embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        img_size=pair(img_size)
        patch_size=pair(patch_size)

        self.img_size=img_size
        self.patch_size=patch_size

        self.H,self.W=img_size[0]//stride , img_size[1]//stride
        num_pathces= self.H * self.W

        self.proj=nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0]//2,patch_size[1]//2)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,x):
        x=self.proj(x)
        H, W =x.shape[2:]

        x=self.norm(x.flatten(2).transpose(1,2))
        return x , H ,W
