import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from ...decoder import MLPDecoder
from ...encoder import OverlapPatchEmbed, PVTEncoder, PVTPosEmbedding
from ...utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class PVTClassification(nn.Module):
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
    embed_dims:  int
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
        Override default qk scale of head_dim ** -0.5 Spatial Attention if set
    p_dropout: float,
        Dropout rate,default is 0.0
    attn_dropout:  float,
        Attention dropout rate, default is 0.0
    drop_path_rate: float
        Stochastic depth rate, default is 0.1
    norm_layer:
        Normalization layer, default is nn.LayerNorm
    sr_ratios: float
        Spatial reduction ratio
    decoder_config:int or tuple[int], optional
        Configuration of the decoder. If None, the default configuration is used.
    linear: bool
        Whether to use linear Spatial attention, default is False
    use_dwconv: bool
        Whether to use Depth-wise convolutions, default is False
    ape: bool
        Whether to use absolute position embedding, default is True
    """

    def __init__(
        self,
        img_size=224,
        patch_size=[7, 3, 3, 3],
        in_channels=3,
        n_classes=1000,
        embed_dims=[64, 128, 256, 512],
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
        decoder_config=None,
        linear=False,
        use_dwconv=False,
        ape=True,
    ):
        super(PVTClassification, self).__init__()
        self.ape = ape
        self.depths = depths
        assert (
            len(depths) == len(num_heads) == len(embed_dims)
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
                            in_channels=in_channels if i == 0 else embed_dims[i - 1],
                            embedding_dim=embed_dims[i],
                        )
                    ]
                )
            )
            if ape:
                if i != len(depths) - 1:
                    self.pos_embeds.append(
                        nn.ModuleList(
                            [
                                PVTPosEmbedding(
                                    pos_shape=img_size // np.prod(patch_size[: i + 1]),
                                    pos_dim=embed_dims[i],
                                )
                            ]
                        )
                    )
                else:
                    self.last_pos = nn.Parameter(
                        torch.randn(
                            1,
                            (img_size // np.prod(patch_size[: i + 1])) ** 2,
                            embed_dims[-1],
                        )
                    )

            self.blocks.append(
                nn.ModuleList(
                    [
                        PVTEncoder(
                            dim=embed_dims[i],
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
            self.norms.append(norm_layer(embed_dims[i]))
        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))
        trunc_normal_(self.cls_token, std=0.02)

        if decoder_config is not None:

            if not isinstance(decoder_config, list) and not isinstance(
                decoder_config, tuple
            ):
                decoder_config = [decoder_config]
            assert (
                decoder_config[0] == embed_dims[-1]
            ), f"Configurations do not match for MLPDecoder, First element of `decoder_config` expected to be {embed_dims[-1]}, got {decoder_config[0]} "
            self.decoder = MLPDecoder(config=decoder_config, n_classes=n_classes)
        else:
            self.decoder = MLPDecoder(config=embed_dims[-1], n_classes=n_classes)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of size `n_classes`

        """
        B = x.shape[0]
        for i in range(len(self.depths)):
            patch_embed = self.patch_embeds[i]
            block = self.blocks[i]
            norm = self.norms[i]
            x, H, W = patch_embed[0](x)
            N = x.shape[1]
            if self.ape:
                if i == len(self.depths) - 1:
                    x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

                    x += self.last_pos[:, : (N + 1)]
                else:
                    pos_embed = self.pos_embeds[i]
                    x = pos_embed[0](x, H=H, W=W)
            for blk in block:
                x = blk(x, H=H, W=W)
            x = norm(x)
            if i == len(self.depths) - 1:
                x = x.mean(dim=1)
            else:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.decoder(x)
        return x


@MODEL_REGISTRY.register()
class PVTClassificationV2(PVTClassification):
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
        Input channels in image, default is 3
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
    norm_layer:nn.Module
        Normalization layer, default is nn.LayerNorm
    sr_ratios: float
        Spatial reduction ratio
    decoder_config:int or tuple[int], optional
        Configuration of the decoder. If None, the default configuration is used.
    linear: bool
        Whether to use linear Spatial attention, default is False
    use_dwconv: bool
        Whether to use Depth-wise convolutions, default is True
    ape: bool
        Whether to use absolute position embedding, default is false
    """

    def __init__(
        self,
        img_size=224,
        patch_size=[7, 3, 3, 3],
        in_channels=3,
        n_classes=1000,
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
        decoder_config=None,
        use_dwconv=True,
        linear=False,
        ape=False,
    ):
        super(PVTClassificationV2, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            n_classes=n_classes,
            embed_dims=embedding_dims,
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
            decoder_config=decoder_config,
            ape=ape,
            use_dwconv=use_dwconv,
            linear=linear,
        )
