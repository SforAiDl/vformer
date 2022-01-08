import types

import torch
import torch.nn as nn

from ...utils.dpt_utils import (
    FeatureFusionBlock_custom,
    Interpolate,
    Transpose,
    _resize_pos_embed,
    forward_flex,
    get_readout_oper,
)
from ...utils.registry import MODEL_REGISTRY

activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


attention = {}


def get_attention(name):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.to_qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook


@MODEL_REGISTRY.register()
class DPTDepth(nn.Module):
    """
    Parameters
    ----------
    backbone:str
    features:int
    readout:str
    hooks: list[int]
    channels_last: bool
    use_bn:bool
    enable_attention_hooks:bool
    non_negative:bool
    scale:float
    shift:float
    invert:bool
    start_index:int
    """
    def __init__(
        self,
        backbone,
        in_channels =3,
        features=256,
        readout="project",
        hooks=(2, 5, 8, 11),
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        non_negative=True,
        scale=1.0,
        shift=0.0,
        invert=False,
        start_index=1
    ):
        super(DPTDepth, self).__init__()
        self.channels_last = channels_last
        self.use_bn = use_bn
        self.enable_attention_hooks = enable_attention_hooks
        self.non_negative = non_negative
        self.scale = scale
        self.shift = shift
        self.features = features
        self.invert = invert

        if backbone == "vitb16_384":
            scratch_in_features = (
                96,
                192,
                384,
                768,
            )
            self.model = MODEL_REGISTRY.get("VanillaViT")(
                img_size=384,
                patch_size=16,
                embedding_dim=768,
                head_dim=64,
                depth=12,
                attn_heads=12,
                encoder_mlp_dim=768,
                n_classes=10,
                in_channels=in_channels,
            )
            hooks = [2, 5, 8, 11] if hooks is None else hooks
            self.vit_features = 768
        elif backbone == "vitl16_384":
            scratch_in_features = (256, 512, 1024, 1024)
            self.model = MODEL_REGISTRY.get("VanillaViT")(
                img_size=384,
                patch_size=16,
                embedding_dim=1024,
                head_dim=64,
                depth=24,
                attn_heads=16,
                encoder_mlp_dim=1024,
                n_classes=10,
                in_channels=in_channels,
            )
            hooks = [5, 11, 17, 23] if hooks is None else hooks
            self.vit_features = 1024
        elif backbone=="vit_tiny":
            scratch_in_features = (48,96,144,192)
            self.model = MODEL_REGISTRY.get("VanillaViT")(
                img_size =384,
                patch_size=16,
                embedding_dim=192,
                head_dim=64,
                depth=12,
                attn_heads =3,
                encoder_mlp_dim = 192,
                n_classes=3, #doenst matter because decoder part is not used in DPTs forward_vit
                in_channels=in_channels
            )
            hooks = [2,5,8,11] if hooks is None else hooks
            self.vit_features = 192
        else:
            raise NotImplementedError

        self._register_hooks_and_add_postprocess(
            features=scratch_in_features,
            hooks=hooks,
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
            start_index=start_index
        )
        self._make_scratch(
            in_shape=scratch_in_features,
            out_shape=features,
            groups=1,
            expand=False,
        )
        self._add_refinenet_to_scratch(features=features, use_bn=use_bn)

        self.model.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

    def _register_hooks_and_add_postprocess(
        self,
        size=(384, 384),
        features=(96, 192, 384, 768),
        hooks=(2, 5, 8, 11),
        use_readout="ignore",
        enable_attention_hooks=False,
        start_index=1,
    ):

        for i in range(4):
            self.model.encoder.encoder[hooks[i]][0].fn.register_forward_hook(
                get_activation(str(i + 1))
            )

        self.activations = activations

        if enable_attention_hooks:
            for i in range(4):
                self.model.encoder.encoder[hooks[i]][0].fn.register_forward_hook(
                    get_attention(f"attn_{str(i+1)}")
                )
            self.attention = attention

        readout_oper = get_readout_oper(
            self.vit_features, features, use_readout, start_index
        )

        # 32, 48, 136, 384

        self.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=self.vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        self.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=self.vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        self.act_postprocess3 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=self.vit_features,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.act_postprocess4 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=self.vit_features,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.model.start_index = start_index
        self.model.patch_size = [16, 16]


        self.model.forward_flex = types.MethodType(forward_flex, self.model)
        self.model._resize_pos_embed = types.MethodType(_resize_pos_embed, self.model)

    def _make_scratch(self, in_shape, out_shape, groups=1, expand=False):
        self.scratch = nn.Module()

        for i in range(4):
            layer = nn.Conv2d(
                in_shape[i],
                out_shape * (2) ** (i) if expand else out_shape,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=groups,
            )
            setattr(self.scratch, f"layer{i+1}_rn", layer)

    def _add_refinenet_to_scratch(self, features, use_bn):

        for i in range(4):
            refinenet = FeatureFusionBlock_custom(
                features,
                nn.ReLU(False),
                deconv=False,
                bn=use_bn,
                expand=False,
                align_corners=True,
            )
            setattr(self.scratch, f"refinenet{i+1}", refinenet)

    def forward_vit(self, x):
        """

        Parameters
        -----------
        """

        b, c, h, w = x.shape

        glob = forward_flex(self, x)

        layer_1 = self.activations["1"]
        layer_2 = self.activations["2"]
        layer_3 = self.activations["3"]
        layer_4 = self.activations["4"]

        layer_1 = self.act_postprocess1[0:2](layer_1)
        layer_2 = self.act_postprocess2[0:2](layer_2)
        layer_3 = self.act_postprocess3[0:2](layer_3)
        layer_4 = self.act_postprocess4[0:2](layer_4)

        unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(
                    [
                        h // self.model.patch_size[1],
                        w // self.model.patch_size[0],
                    ]
                ),
            )
        )

        if layer_1.ndim == 3:
            layer_1 = unflatten(layer_1)
        if layer_2.ndim == 3:
            layer_2 = unflatten(layer_2)
        if layer_3.ndim == 3:
            layer_3 = unflatten(layer_3)
        if layer_4.ndim == 3:
            layer_4 = unflatten(layer_4)

        layer_1 = self.act_postprocess1[3 : len(self.act_postprocess1)](layer_1)
        layer_2 = self.act_postprocess2[3 : len(self.act_postprocess2)](layer_2)
        layer_3 = self.act_postprocess3[3 : len(self.act_postprocess3)](layer_3)
        layer_4 = self.act_postprocess4[3 : len(self.act_postprocess4)](layer_4)

        return layer_1, layer_2, layer_3, layer_4

    def forward(self, x):
        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = self.forward_vit(x)

        layer_1 = self.scratch.layer1_rn(layer_1)
        layer_2 = self.scratch.layer2_rn(layer_2)
        layer_3 = self.scratch.layer3_rn(layer_3)
        layer_4 = self.scratch.layer4_rn(layer_4)

        path1 = self.scratch.refinenet4(layer_4)
        path1 = self.scratch.refinenet3(path1, layer_3)
        path1 = self.scratch.refinenet2(path1, layer_2)
        path1 = self.scratch.refinenet1(path1, layer_1)

        inv_depth = self.model.head(path1).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth
