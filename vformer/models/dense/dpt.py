import types

import torch
import torch.nn as nn

from ...utils.dpt_utils import FeatureFusionBlock_custom, Transpose, get_readout_oper
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
            module.qkv(x)
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


class DPTdept(nn.Module):
    def __init__(
        self,
        backbone,
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
    ):
        super(DPTdept, self).__init__()
        self.channels_last = channels_last
        self.use_bn = use_bn
        self.enable_attention_hooks = enable_attention_hooks
        self.non_negative = non_negative
        self.scale = scale
        self.shift = shift
        self.features = features
        self.invert = invert

        scratch_in_features = (
            96,
            192,
            384,
            768,
        )  # only two values possible, based on pretrain string we will have one if -else block to handle this
        features_dict = {}  # these will contain the parameters for vitl16 and vitb16
        self.model = MODEL_REGISTRY.get(backbone)(**features_dict)
        self._register_hooks_and_add_postprocess(
            features=scratch_in_features,
            hooks=hooks,
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        self._make_scratch(
            in_shape=(96, 192, 384, 768),
            out_shape=self.features,
            groups=1,
            expand=False,
        )
        self._add_refinenet_to_scratch(features=self.features, use_bn=self.use_bn)

    def _register_hooks_and_add_postprocess(
        self,
        size=(384, 384),
        features=(96, 192, 384, 768),
        hooks=(2, 5, 8, 11),
        use_readout="ignore",
        enable_attention_hooks=False,
        vit_features=768,
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

        readout_oper = get_readout_oper(
            vit_features, features, use_readout, start_index
        )

        # 32, 48, 136, 384

        self.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
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
                in_channels=vit_features,
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
                in_channels=vit_features,
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
                in_channels=vit_features,
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

        # We inject this function into the VisionTransformer instances so that
        # we can use it with interpolated position embeddings without modifying the library source.
        self.model.forward_flex = types.MethodType(self.forward_flex, self.model)
        self.model._resize_pos_embed = types.MethodType(
            self._resize_pos_embed, self.model
        )

    def _make_scratch(self, in_shape, out_shape, groups=1, expand=False):
        if isinstance(in_shape, int):
            in_shape = [in_shape] * 4
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
            setattr(self.scratch, f"refinet{i}", refinenet)

    def forward_vit(self, x):
        """

        Parameters
        -----------
        """

        b, c, h, w = x.shape

        glob = self.model.forward_flex(x)

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

    def forward_flex(self, x):
        b, c, h, w = x.shape

        pos_embed = self._resize_pos_embed(
            self.pos_embed.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
        )

        B = x.shape[0]

        if hasattr(self.patch_embed, "backbone"):
            x = self.patch_embed.backbone(x)
            if isinstance(x, (list, tuple)):
                x = x[-1]  # last feature if backbone outputs list/tuple of features

            x = self.patch_embed.patch_embedding(x)
        if getattr(self, "dist_token", None) is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_embed.pos_drop(x)
        x = self.encoder(x)
        return x

    def forward(self, x):
        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = self.forward_vit(self.model, x)

        layer_1 = self.scratch.layer1_rn(layer_1)
        layer_2 = self.scratch.layer1_rn(layer_2)
        layer_3 = self.scratch.layer1_rn(layer_3)
        layer_4 = self.scratch.layer1_rn(layer_4)

        path1 = self.scratch.refinenet4(layer_4)
        path1 = self.scratch.refinenet3(path1, layer_3)
        path1 = self.scratch.refinenet2(path1, layer_2)
        path1 = self.scratch.refinenet1(path1, layer_1)

        inv_depth = self.scratch.output_conv(path1).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth
