import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from ...encoder.convvt import ConvVTStage
from ...utils import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ConvVT(nn.Module):
    """
    Implementation of CvT: Introducing Convolutions to Vision Transformers:
    https://arxiv.org/pdf/2103.15808.pdf

    Parameters:
    ------------
    img_size: int
        Size of the image, default is 224
    in_channels:int
        Number of input channels in image, default is 3
    num_stages: int
        Number of stages in encoder block, default is 3
    n_classes: int
        Number of classes for classification, default is 1000
    * The following are all in list of int/float with length num_stages
    patch_size: list[int]
        Size of patch, default is [7, 3, 3]
    patch_stride: list[int]
        Stride of patch, default is [4, 2, 2]
    patch_padding: list[int]
        Padding for patch, default is [2, 1, 1]
    embedding_dim: list[int]
        Embedding dimensions, default is [64, 192, 384]
    depth: list[int]
        Number of CVT Attention blocks in each stage, default is [1, 2, 10]
    num_heads: list[int]
        Number of heads in attention, default is [1, 3, 6]
    mlp_ratio: list[float]
        Feature dimension expansion ratio in MLP, default is [4.0, 4.0, 4.0]
    p_dropout: list[float]
        Probability of dropout in MLP, default is [0, 0, 0]
    attn_dropout: list[float]
        Probability of dropout in attention, default is [0, 0, 0]
    drop_path_rate: list[float]
        Probability for droppath, default is [0, 0, 0.1]
    kernel_size: list[int]
        Size of kernel, default is [3, 3, 3]
    padding_q: list[int]
        Size of padding in q, default is [1, 1, 1]
    padding_kv: list[int]
        Size of padding in kv, default is [2, 2, 2]
    stride_kv: list[int]
        Stride in kv, default is [2, 2, 2]
    stride_q: list[int]
        Stride in q, default is [1, 1, 1]
    """

    def __init__(
        self,
        img_size=224,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        embedding_dim=[64, 192, 384],
        num_heads=[1, 3, 6],
        depth=[1, 2, 10],
        mlp_ratio=[4.0, 4.0, 4.0],
        p_dropout=[0, 0, 0],
        attn_dropout=[0, 0, 0],
        drop_path_rate=[0, 0, 0.1],
        kernel_size=[3, 3, 3],
        padding_q=[1, 1, 1],
        padding_kv=[1, 1, 1],
        stride_kv=[2, 2, 2],
        stride_q=[1, 1, 1],
        in_channels=3,
        num_stages=3,
        n_classes=1000,
    ):
        super().__init__()

        self.n_classes = n_classes

        self.num_stages = num_stages
        self.stages = []
        for i in range(self.num_stages):
            stage = ConvVTStage(
                in_channels=in_channels,
                img_size=img_size // (4 * 2 ** i),
                with_cls_token=False if i < self.num_stages - 1 else True,
                patch_size=patch_size[i],
                patch_stride=patch_stride[i],
                patch_padding=patch_padding[i],
                embedding_dim=embedding_dim[i],
                depth=depth[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio[i],
                p_dropout=p_dropout[i],
                attn_dropout=attn_dropout[i],
                drop_path_rate=drop_path_rate[i],
                kernel_size=kernel_size[i],
                padding_q=padding_q[i],
                padding_kv=padding_kv[i],
                stride_kv=stride_kv[i],
                stride_q=stride_q[i],
            )
            self.stages.append(stage)
            in_channels = embedding_dim[i]

        self.norm = nn.LayerNorm(embedding_dim[-1])

        # Classifier head
        self.head = (
            nn.Linear(embedding_dim[-1], n_classes) if n_classes > 0 else nn.Identity()
        )
        trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x):

        for i in range(self.num_stages):
            x, cls_tokens = self.stages[i](x)

        x = self.norm(cls_tokens)
        x = torch.squeeze(x)
        x = self.head(x)

        return x
