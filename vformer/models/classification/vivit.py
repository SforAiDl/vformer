import torch
import torch.nn as nn
from einops import rearrange, repeat

from ...common.base_model import BaseClassificationModel
from ...decoder.mlp import MLPDecoder
from ...encoder.embedding import LinearVideoEmbedding, PosEmbedding, TubeletEmbedding
from ...encoder.vanilla import VanillaEncoder
from ...encoder.vivit import ViViTEncoder
from ...utils.registry import MODEL_REGISTRY
from ...utils.utils import pair


@MODEL_REGISTRY.register()
class ViViTModel2(BaseClassificationModel):
    """
    Model 2 implementation of  A Video vision Transformer -
    Parameters
    ----------
    img_size:int
        Size of single frame/ image in video
    in_channels:int
        Number of channels
    patch_size: int
        Patch size
    embedding_dim: int
        Embedding dimension of a patch
    num_frames:int
        Number of seconds in each Video
    depth:int
        Number of encoder layers
    num_heads:int
        Number of attention heads
    head_dim:int
        Dimension of head
    num_classes:int
        Number of classes
    mlp_dim: int
        Dimension of hidden layer
    pool: str
        Pooling operation,must be one of {"cls","mean"},default is "cls"
    p_dropout:float
        Dropout probability
    attn_dropout:float
        Dropout probability
    drop_path_rate:float
        Stochastic drop path rate
    """

    def __init__(
        self,
        img_size,
        in_channels,
        patch_size,
        embedding_dim,
        num_frames,
        depth,
        num_heads,
        head_dim,
        num_classes,
        mlp_dim=None,
        pool="cls",
        p_dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.02,
    ):
        super(ViViTModel2, self).__init__(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            pool=pool,
        )

        patch_dim = in_channels * patch_size ** 2
        self.patch_embedding = LinearVideoEmbedding(
            embedding_dim=embedding_dim,
            patch_height=patch_size,
            patch_width=patch_size,
            patch_dim=patch_dim,
        )

        self.pos_embedding = PosEmbedding(
            shape=[num_frames, self.num_patches + 1], dim=embedding_dim, drop=p_dropout
        )

        self.space_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )  # this is similar to using cls token in vanilla vision transformer
        self.spatial_transformer = VanillaEncoder(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            p_dropout=p_dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
        )

        self.time_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.temporal_transformer = VanillaEncoder(
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            p_dropout=p_dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
        )

        self.decoder = MLPDecoder(
            config=[
                embedding_dim,
            ],
            n_classes=num_classes,
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        (
            b,
            t,
            n,
            d,
        ) = x.shape  # shape of x will be number of videos,time,num_frames,embedding dim
        cls_space_tokens = repeat(self.space_token, "() n d -> b t n d", b=b, t=t)

        x = nn.Parameter(torch.cat((cls_space_tokens, x), dim=2))
        x = self.pos_embedding(x)

        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.spatial_transformer(x)
        x = rearrange(x[:, 0], "(b t) ... -> b t ...", b=b)

        cls_temporal_tokens = repeat(self.time_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.decoder(x)

        return x


# model 3
@MODEL_REGISTRY.register()
class ViViTModel3(BaseClassificationModel):
    """
    model 3 of A video Vision Trnasformer- https://arxiv.org/abs/2103.15691

    Parameters
    ----------
    img_size:int or tuple[int]
        size of a frame
    patch_t:int
        Temporal length of single tube/patch in tubelet embedding
    patch_h:int
        Height  of single tube/patch in tubelet embedding
    patch_w:int
        Width  of single tube/patch in tubelet embedding
    in_channels: int
        Number of input channels, default is 3
    num_classes:int
        Number of classes
    num_frames :int
        Number of seconds in each Video
    embedding_dim:int
        Embedding dimension of a patch
    depth:int
        Number of Encoder layers
    num_heads: int
        Number of attention heads
    head_dim:int
        Dimension of attention head
    p_dropout:float
        Dropout rate/probability, default is 0.0
    mlp_dim: int
        Hidden dimension, optional
    """

    def __init__(
        self,
        img_size,
        patch_t,
        patch_h,
        patch_w,
        in_channels,
        num_classes,
        num_frames,
        embedding_dim,
        depth,
        num_heads,
        head_dim,
        p_dropout,
        mlp_dim=None,
    ):

        super(ViViTModel3, self).__init__(
            in_channels=in_channels,
            patch_size=(patch_h, patch_w),
            pool="mean",
            img_size=img_size,
        )
        h, w = pair(img_size)
        self.tubelet_embedding = TubeletEmbedding(
            embedding_dim=embedding_dim,
            tubelet_t=patch_t,
            tubelet_h=patch_h,
            tubelet_w=patch_w,
            in_channels=in_channels,
        )

        self.pos_embbedding = PosEmbedding(
            shape=[num_frames // patch_t, (h * w) // (patch_w * patch_h)],
            dim=embedding_dim,
        )
        self.encoder = ViViTEncoder(
            dim=embedding_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            p_dropout=p_dropout,
            depth=depth,
            hidden_dim=mlp_dim,
        )

        self.decoder = MLPDecoder(
            config=[
                embedding_dim,
            ],
            n_classes=num_classes,
        )

    def forward(self, x):

        x = self.tubelet_embedding(x)
        x = self.pos_embbedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.decoder(x)

        return x
