import einops
import torch.nn as nn

from ...attention import VanillaSelfAttention
from ...encoder.embedding.pos_embedding import PosEmbedding
from ...utils import ATTENTION_REGISTRY, MODEL_REGISTRY


class VisformerConvBlock(nn.Module):
    """
    Convolution Block for Vision-Friendly transformers
    https://arxiv.org/abs/2104.12533

    Parameters
    ----------
    in_channels: int
        Number of input channels
    group: int
        Number of groups for convolution, default is 8
    activation: torch.nn.Module
        Activation function between layers, default is nn.GELU
    p_dropout: float
        Dropout rate, default is 0.0
    """

    def __init__(self, in_channels, group=8, activation=nn.GELU, p_dropout=0.0):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=3,
            padding=1,
            groups=group,
            bias=False,
        )
        self.act2 = activation()
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of same size as input
        """

        xt = x
        xt = self.norm1(xt)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = x + xt

        return x


@ATTENTION_REGISTRY.register()
class VisformerAttentionBlock(nn.Module):
    """
    Attention Block for Vision-Friendly transformers
    https://arxiv.org/abs/2104.12533

    Parameters
    ----------
    in_channels: int
        Number of input channels
    num_heads: int
        Number of heads for attention, default is 8
    activation: torch.nn.Module
        Activation function between layers, default is nn.GELU
    p_dropout: float
        Dropout rate, default is 0.0
    """

    def __init__(self, in_channels, num_heads=8, activation=nn.GELU, p_dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, bias=False)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False)
        self.attn = VanillaSelfAttention(
            in_channels, num_heads=num_heads, head_dim=in_channels // num_heads
        )
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        Returns
        ----------
        torch.Tensor
            Returns tensor of same size as input
        """

        B, C, H, W = x.shape
        xt = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.norm1(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.attn(x)
        x = xt + x
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        xt = x
        x = self.norm2(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = xt + x

        return x


@MODEL_REGISTRY.register()
class Visformer(nn.Module):
    """
    A builder to construct a Vision-Friendly transformer model as in the paper: "Visformer: A vision-friendly transformer"
    https://arxiv.org/abs/2104.12533

    Parameters
    ----------
    img_size: int,tuple
        Size of the input image
    n_classes: int
        Number of classes in the dataset
    depth: tuple[int]
        Number of layers before each embedding reduction
    config: tuple[int]
        Choice of convolution block (0) or attention block (1) for corresponding layer
    channel_config: tuple[int]
        Number of channels for each layer
    num_heads: int
        Number of heads for attention block, default is 8
    conv_group: int
        Number of groups for convolution block, default is 8
    p_dropout_conv: float
        Dropout rate for convolution block, default is 0.0
    p_dropout_attn: float
        Dropout rate for attention block, default is 0.0
    activation: torch.nn.Module
        Activation function between layers, default is nn.GELU
    pos_embedding: bool
        Whether to use positional embedding, default is True

    """

    def __init__(
        self,
        img_size,
        n_classes,
        depth: tuple,
        config: tuple,
        channel_config: tuple,
        num_heads=8,
        conv_group=8,
        p_dropout_conv=0.0,
        p_dropout_attn=0.0,
        activation=nn.GELU,
        pos_embedding=True,
    ):
        super().__init__()

        q = 0
        assert (
            len(channel_config) == len(depth) - depth.count(0) + 2
        ), "Channel config is not correct"

        assert set(config).issubset(
            set([0, 1])
        ), "Config is not correct, should contain only 0 and 1"

        self.linear = nn.Linear(channel_config[-1], n_classes)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        image_size = list(img_size)

        assert image_size[0] // (2 ** (len(depth) + 1)) > 0, "Image size is too small"
        assert image_size[1] // (2 ** (len(depth) + 1)) > 0, "Image size is too small"

        self.stem = nn.ModuleList(
            [
                nn.Conv2d(
                    channel_config[q],
                    channel_config[q + 1],
                    kernel_size=7,
                    padding=3,
                    stride=2,
                    bias=False,
                ),
                nn.BatchNorm2d(channel_config[q + 1]),
                nn.ReLU(inplace=True),
            ]
        )

        q += 1
        emb = 2
        image_size = [i // 2 for i in image_size]

        for i in range(len(depth)):

            if depth[i] == 0:
                emb *= 2
                config = tuple([0] + list(config))
                continue

            self.stem.extend(
                [
                    nn.Conv2d(
                        channel_config[q],
                        channel_config[q + 1],
                        kernel_size=emb,
                        stride=emb,
                    ),
                    nn.BatchNorm2d(channel_config[q + 1]),
                    nn.ReLU(inplace=True),
                ]
            )

            image_size = [k // emb for k in image_size]
            emb = 2
            q += 1

            if pos_embedding:
                self.stem.extend(
                    [PosEmbedding([channel_config[q], image_size[0]], image_size[1])]
                )

            if config[i] == 0:
                self.stem.extend(
                    [
                        VisformerConvBlock(
                            channel_config[q],
                            group=conv_group,
                            p_dropout=p_dropout_conv,
                            activation=activation,
                        )
                        for j in range(depth[i])
                    ]
                )

            elif config[i] == 1:
                self.stem.extend(
                    [
                        VisformerAttentionBlock(
                            channel_config[q],
                            num_heads,
                            activation,
                            p_dropout_attn,
                        )
                        for j in range(depth[i])
                    ]
                )

        self.stem.extend([nn.BatchNorm2d(channel_config[-1]), nn.AdaptiveAvgPool2d(1)])
        self.softmax = nn.Softmax(dim=1)

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
        for i in self.stem:
            x = i(x)

        x.squeeze_(2).squeeze_(2)
        x = self.linear(x)
        x = self.softmax(x)

        return x


@MODEL_REGISTRY.register()
def Visformer_S(img_size, n_classes, in_channels=3):
    """
    Visformer-S model from the paper:"Visformer: The Vision-friendly Transformer"
    https://arxiv.org/abs/1906.11488

    Parameters
    ----------
    img_size: int,tuple
        Size of the input image
    n_classes: int
        Number of classes in the dataset
    in_channels: int
        Number of channels in the input
    """
    return Visformer(
        img_size,
        n_classes,
        (0, 7, 4, 4),
        (0, 1, 1),
        (in_channels, 32, 192, 384, 768),
        num_heads=6,
    )


@MODEL_REGISTRY.register()
def VisformerV2_S(img_size, n_classes, in_channels=3):
    """
    VisformerV2-S model from the paper:"Visformer: The Vision-friendly Transformer"
    https://arxiv.org/abs/1906.11488

    Parameters
    ----------
    img_size: int,tuple
        Size of the input image
    n_classes: int
        Number of classes in the dataset
    in_channels: int
        Number of channels in the input
    """
    return Visformer(
        img_size,
        n_classes,
        (1, 10, 14, 3),
        (0, 0, 1, 1),
        (in_channels, 32, 64, 128, 256, 512),
        num_heads=6,
    )


@MODEL_REGISTRY.register()
def Visformer_Ti(img_size, n_classes, in_channels=3):
    """
    Visformer-Ti model from the paper:"Visformer: The Vision-friendly Transformer"
    https://arxiv.org/abs/1906.11488

    Parameters
    ----------
    img_size: int,tuple
        Size of the input image
    n_classes: int
        Number of classes in the dataset
    in_channels: int
        Number of channels in the input
    """
    return Visformer(
        img_size,
        n_classes,
        (0, 7, 4, 4),
        (0, 1, 1),
        (in_channels, 16, 96, 192, 384),
        num_heads=6,
    )


@MODEL_REGISTRY.register()
def VisformerV2_Ti(img_size, n_classes, in_channels=3):
    """
    VisformerV2-Ti model from the paper:"Visformer: The Vision-friendly Transformer"
    https://arxiv.org/abs/1906.11488

    Parameters
    ----------
    img_size: int,tuple
        Size of the input image
    n_classes: int
        Number of classes in the dataset
    in_channels: int
        Number of channels in the input
    """

    return Visformer(
        img_size,
        n_classes,
        (1, 4, 6, 2),
        (0, 0, 1, 1),
        (in_channels, 24, 48, 96, 192, 384),
        num_heads=6,
    )
