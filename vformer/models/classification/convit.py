import torch
from einops import repeat

from ...encoder import ConViTEncoder
from ...utils import MODEL_REGISTRY
from .vanilla import VanillaViT


@MODEL_REGISTRY.register()
class ConViT(VanillaViT):
    """
    Implementation of 'ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases'
    https://arxiv.org/abs/2103.10697

    Parameters
    ----------
    img_size: int
        Size of the image
    patch_size: int
        Size of a patch
    n_classes: int
        Number of classes for classification
    embedding_dim: int
        Dimension of hidden layer
    head_dim: int
        Dimension of the attention head
    depth_sa: int
        Number of attention layers in the encoder for self attention layers
    depth_gpsa: int
        Number of attention layers in the encoder for global positional self attention layers
    attn_heads_sa:int
        Number of the attention heads for self attention layers
    attn_heads_gpsa:int
        Number of the attention heads for global positional self attention layers
    encoder_mlp_dim: int
        Dimension of hidden layer in the encoder
    in_channels: int
        Number of input channels
    decoder_config: int or tuple or list, optional
        Configuration of the decoder. If None, the default configuration is used.
    pool: {"cls","mean"}
        Feature pooling type
    p_dropout_encoder: float
        Dropout probability in the encoder
    p_dropout_embedding: float
        Dropout probability in the embedding layer
    """

    def __init__(
        self,
        img_size,
        patch_size,
        n_classes,
        embedding_dim=1024,
        head_dim=64,
        depth_sa=6,
        depth_gpsa=6,
        attn_heads_sa=16,
        attn_heads_gpsa=16,
        encoder_mlp_dim=2048,
        in_channels=3,
        decoder_config=None,
        pool="cls",
        p_dropout_encoder=0,
        p_dropout_embedding=0,
    ):
        super().__init__(
            img_size,
            patch_size,
            n_classes,
            embedding_dim,
            head_dim,
            depth_sa,
            attn_heads_sa,
            encoder_mlp_dim,
            in_channels,
            decoder_config,
            pool,
            p_dropout_encoder,
            p_dropout_embedding,
        )

        self.encoder_gpsa = ConViTEncoder(
            embedding_dim,
            depth_gpsa,
            attn_heads_gpsa,
            head_dim,
            encoder_mlp_dim,
            p_dropout_encoder,
        )

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
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x_cls = x[:, 0:1, :]
        x = x[:, 1:, :]
        x = self.encoder_gpsa(x)
        x = torch.cat((x_cls, x), dim=1)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.decoder(x)

        return x
