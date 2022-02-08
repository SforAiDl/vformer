import numpy as np
import torch
import torch.nn as nn

from ....encoder import OverlapPatchEmbed, PVTEncoder, PVTPosEmbedding
from ....utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class PVTDetection(nn.Module):
    """
    Implementation of Pyramid Vision Transformer:
    https://arxiv.org/abs/2102.12122v1


    Parameters
    ----------
    img_size: int
        Image size
    patch_size: list(int)
        List of patch size
    in_channels: int
        Input channels in image, default=3
    n_classes: int
        Number of classes for classification
    embedding_dims:  int
        Patch Embedding dimension
    num_heads:tuple[int]
        Number of heads in each transformer layer
    depths: tuple[int]
        Depth in each Transformer layer
    mlp_ratio: float
        Ratio of mlp heads to embedding dimension
    qkv_bias: bool, default= True
        Adds bias to the qkv if true
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 in Spatial Attention if set
    p_dropout: float,
        Dropout rate,default is 0.0
    attn_dropout:  float,
        Attention dropout rate, default is 0.0
    drop_path_rate: float
        Stochastic depth rate, default is 0.1
    sr_ratios: float
        Spatial reduction ratio
    linear: bool
        Whether to use linear spatial attention, default is False
    use_dwconv: bool
        Whether to use Depth-wise convolutions in Overlap-patch embedding, default is False
    ape: bool
        Whether to use absolute position embedding, default is True

    """

    def __init__(
        self,
        img_size=224,
        patch_size=[7, 3, 3, 3],
        in_channels=3,
        embedding_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratio=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        p_dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        linear=False,
        use_dwconv=False,
        ape=True,
    ):
        super(PVTDetection, self).__init__()

        self.ape = ape
        self.depths = depths

        assert (
            len(depths) == len(num_heads) == len(embedding_dims)
        ), "Configurations do not match"

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embeds = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        self.norms = nn.ModuleList()
        self.pos_embeds = nn.ModuleList()

        for i in range(len(depths)):

            self.patch_embeds.append(
                nn.ModuleList(
                    [
                        OverlapPatchEmbed(
                            img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                            patch_size=patch_size[i],
                            stride=4 if i == 0 else 2,
                            in_channels=in_channels
                            if i == 0
                            else embedding_dims[i - 1],
                            embedding_dim=embedding_dims[i],
                        )
                    ]
                )
            )

            if ape:
                self.pos_embeds.append(
                    nn.ModuleList(
                        [
                            PVTPosEmbedding(
                                pos_shape=img_size // np.prod(patch_size[: i + 1]),
                                pos_dim=embedding_dims[i],
                            )
                        ]
                    )
                )

            self.blocks.append(
                nn.ModuleList(
                    [
                        PVTEncoder(
                            dim=embedding_dims[i],
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio[i],
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            p_dropout=p_dropout,
                            depth=depths[i],
                            attn_dropout=attn_dropout,
                            drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                            sr_ratio=sr_ratios[i],
                            linear=linear,
                            act_layer=nn.GELU,
                            use_dwconv=use_dwconv,
                        )
                    ]
                )
            )
            self.norms.append(norm_layer(embedding_dims[i]))

        self.pool = nn.Parameter(torch.zeros(1, 1, embedding_dims[-1]))

    def forward(self, x):

        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns list containing output features from all pyramid stages

        """
        B = x.shape[0]
        out = []

        for i in range(len(self.depths)):

            patch_embed = self.patch_embeds[i]
            block = self.blocks[i]
            norm = self.norms[i]

            x, H, W = patch_embed[0](x)

            if self.ape:
                pos_embed = self.pos_embeds[i]
                x = pos_embed[0](x, H=H, W=W)

            for blk in block:
                x = blk(x, H=H, W=W)

            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            out.append(x)

        return out


@MODEL_REGISTRY.register()
class PVTDetectionV2(PVTDetection):
    """
    Implementation of Pyramid Vision Transformer:
    https://arxiv.org/abs/2102.12122v2


    Parameters
    ----------
    img_size: int
        Image size
    patch_size: list(int)
        List of patch size
    in_channels: int
        Input channels in image, default=3
    n_classes: int
        Number of classes for classification
    embedding_dims:  int
        Patch Embedding dimension
    num_heads:tuple[int]
        Number of heads in each transformer layer
    depths: tuple[int]
        Depth in each Transformer layer
    mlp_ratio: float
        Ratio of mlp heads to embedding dimension
    qkv_bias: bool, default= True
        Adds bias to the qkv if true
    qk_scale: float, optional
        Override default qk scale of head_dim ** -0.5 in Spatial Attention if set
    p_dropout: float,
        Dropout rate,default is 0.0
    attn_dropout:  float,
        Attention dropout rate, default is 0.0
    drop_path_rate: float
        Stochastic depth rate, default is 0.1
    sr_ratios: float
        Spatial reduction ratio
    linear: bool
        Whether to use linear spatial attention
    use_dwconv: bool
        Whether to use Depth-wise convolutions in Overlap-patch embedding
    ape: bool
        Whether to use absolute position embedding
    """

    def __init__(
        self,
        img_size=224,
        patch_size=[7, 3, 3, 3],
        in_channels=3,
        embedding_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratio=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=0.0,
        p_dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        ape=False,
        use_dwconv=True,
        linear=False,
    ):
        super(PVTDetectionV2, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dims=embedding_dims,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            p_dropout=p_dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            depths=depths,
            sr_ratios=sr_ratios,
            linear=linear,
            ape=ape,
            use_dwconv=use_dwconv,
        )
