import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common import BaseClassificationModel
from ...decoder import MLPDecoder
from ...encoder import CVTEmbedding, PosEmbedding, VanillaEncoder
from ...utils import MODEL_REGISTRY, pair


@MODEL_REGISTRY.register()
class CVT(BaseClassificationModel):
    """
    Implementation of Escaping the Big Data Paradigm with Compact Transformers:
    https://arxiv.org/abs/2104.05704

    Parameters:
    ------------
    img_size: int
        Size of the image, default is 224
    patch_size:int
        Size of the single patch in the image, default is 4
    in_channels:int
        Number of input channels in image, default is 3
    seq_pool:bool
        Whether to use sequence pooling, default is True
    embedding_dim: int
        Patch embedding dimension, default is 768
    num_layers: int
        Number of Encoders in encoder block, default is 1
    num_heads: int
        Number of heads in each transformer layer, default is 1
    mlp_ratio:float
        Ratio of mlp heads to embedding dimension, default is 4.0
    n_classes: int
        Number of classes for classification, default is 1000
    p_dropout: float
        Dropout probability, default is 0.0
    attn_dropout: float
        Dropout probability, defualt is 0.0
    drop_path: float
        Stochastic depth rate, default is 0.1
    positional_embedding: str
        One of the string values {'learnable','sine','None'}, default is learnable
    decoder_config: tuple(int) or int
        Configuration of the decoder. If None, the default configuration is used.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_channels=3,
        seq_pool=True,
        embedding_dim=768,
        head_dim=96,
        num_layers=1,
        num_heads=1,
        mlp_ratio=4.0,
        n_classes=1000,
        p_dropout=0.1,
        attn_dropout=0.1,
        drop_path=0.1,
        positional_embedding="learnable",
        decoder_config=(
            768,
            1024,
        ),
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
        )

        assert (
            img_size % patch_size == 0
        ), f"Image size ({img_size}) has to be divisible by patch size ({patch_size})"

        img_size = pair(img_size)
        self.in_channels = in_channels
        self.embedding = CVTEmbedding(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            max_pool=False,
            activation=None,
            num_conv_layers=1,
            conv_bias=True,
        )

        positional_embedding = (
            positional_embedding
            if positional_embedding in ["sine", "learnable", "none"]
            else "sine"
        )
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = self.embedding.sequence_length(
            n_channels=in_channels, height=img_size[0], width=img_size[1]
        )
        self.seq_pool = seq_pool

        assert (
            self.sequence_length is not None or positional_embedding == "none"
        ), f"Positional embedding is set to {positional_embedding} and the sequence length was not specified."

        if not seq_pool:
            self.sequence_length += 1
            self.class_emb = nn.Parameter(
                torch.zeros(1, 1, self.embedding_dim), requires_grad=True
            )
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding != "none":
            self.positional_emb = PosEmbedding(
                shape=self.sequence_length,
                dim=embedding_dim,
                drop=p_dropout,
                sinusoidal=True if positional_embedding is "sine" else False,
            )
        else:
            self.positional_emb = None

        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.encoder_blocks = nn.ModuleList(
            [
                VanillaEncoder(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    depth=1,
                    mlp_dim=hidden_dim,
                    head_dim=head_dim,
                    p_dropout=p_dropout,
                    attn_dropout=attn_dropout,
                    drop_path_rate=dpr[i],
                )
                for i in range(num_layers)
            ]
        )
        if decoder_config is not None:

            if not isinstance(decoder_config, list) and not isinstance(
                decoder_config, tuple
            ):
                decoder_config = [decoder_config]
            assert (
                decoder_config[0] == embedding_dim
            ), f"Configurations do not match for MLPDecoder, First element of `decoder_config` expected to be {embedding_dim}, got {decoder_config[0]} "
            self.decoder = MLPDecoder(config=decoder_config, n_classes=n_classes)
        else:
            self.decoder = MLPDecoder(config=embedding_dim, n_classes=n_classes)

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

        x = self.embedding(x)

        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(
                x, (0, 0, 0, self.in_channels - x.size(1)), mode="constant", value=0
            )

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x = self.positional_emb(x)

        for blk in self.encoder_blocks:
            x = blk(x)

        if self.seq_pool:
            x = torch.matmul(
                F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x
            ).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.decoder(x)

        return x
