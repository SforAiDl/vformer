import types

import torch
import torch.nn as nn

from ...utils.dpt_utils import _resize_pos_embed, forward_flex
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
    Implementation of " Vision Transformers for Dense Prediction "
    https://arxiv.org/abs/2103.13413

    Parameters
    ----------
    backbone:str
        Name of ViT model to be used as backbone, must be one of {`vitb16`,`vitl16`,`vit_tiny`}
    in_channels: int
        Number of channels in input image, default is 3
    img_size: tuple[int]
        Input image size, default is (384,384)
    readout:str
        Method to handle the `readout_token` or `cls_token`
        Must be one of {`add`, `ignore`,`project`}, default is `project`
    hooks: list[int]
        List representing index of encoder blocks on which hooks will be registered.
        These hooks extract features from different ViT blocks, eg attention, default is (2,5,8,11).
    channels_last: bool
        Alters the memory format of storing tensors, default is False,
        For more information visit, this `blogpost<https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`
    use_bn:bool
        If True, BatchNormalisation is used in `FeatureFusionBlock_custom`, default is False
    enable_attention_hooks:bool
        If True, `get_attention` hook is registered, default is false
    non_negative:bool
        If True, Relu operation will be applied in `DPTDepth.model.head` block, default is True
    invert:bool
        If True, forward pass output of `DPTDepth.model.head` will be transformed (inverted)
        according to `scale` and `shift` parameters, default is False
    scale:float
        Float value that will be multiplied with forward pass output from `DPTDepth.model.head`, default is 1.0
    shift:float
        Float value that will be added with forward pass output from `DPTDepth.model.head` after scaling, default is 0.0
    """

    def __init__(
        self,
        backbone,
        in_channels=3,
        img_size=(384, 384),
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
        super(DPTDepth, self).__init__()

        self.channels_last = channels_last
        self.use_bn = use_bn
        self.enable_attention_hooks = enable_attention_hooks
        self.non_negative = non_negative
        self.scale = scale
        self.shift = shift
        self.invert = invert
        start_index = 1

        if backbone == "vitb16":

            scratch_in_features = (
                96,
                192,
                384,
                768,
            )
            self.model = MODEL_REGISTRY.get("VanillaViT")(
                img_size=img_size,
                patch_size=16,
                embedding_dim=768,
                head_dim=64,
                depth=12,
                attn_heads=12,
                encoder_mlp_dim=768,
                n_classes=10,
                in_channels=in_channels,
            )
            hooks = (2, 5, 8, 11) if hooks is None else hooks
            self.vit_features = 768

        elif backbone == "vitl16":

            scratch_in_features = (256, 512, 1024, 1024)
            self.model = MODEL_REGISTRY.get("VanillaViT")(
                img_size=img_size,
                patch_size=16,
                embedding_dim=1024,
                head_dim=64,
                depth=24,
                attn_heads=16,
                encoder_mlp_dim=1024,
                n_classes=10,
                in_channels=in_channels,
            )
            hooks = (5, 11, 17, 23) if hooks is None else hooks
            self.vit_features = 1024

        elif backbone == "vit_tiny":

            scratch_in_features = (48, 96, 144, 192)
            self.model = MODEL_REGISTRY.get("VanillaViT")(
                img_size=img_size,
                patch_size=16,
                embedding_dim=192,
                head_dim=64,
                depth=12,
                attn_heads=3,
                encoder_mlp_dim=192,
                n_classes=3,  # doenst matter because decoder part is not used in DPTs forward_vit
                in_channels=in_channels,
            )
            hooks = (2, 5, 8, 11) if hooks is None else hooks
            self.vit_features = 192

        else:
            raise NotImplementedError

        assert readout in (
            "add",
            "ignore",
            "project",
        ), f"Not valid `readout` param, Must be one of ('add','ignore','project'), but got {readout}"  # check if valid string
        features = scratch_in_features[0]

        self._register_hooks_and_add_postprocess(
            size=img_size,
            features=scratch_in_features,
            hooks=hooks,
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
            start_index=start_index,
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
        """
        Registers forward hooks to the backbone and initializes activation-postprocessing-blocks (act_postprocess(int))
        Parameters
        ----------
        size: tuple[int]
            Input image size
        features:tuple[int]
            Number of features
        hooks:tuple[int]
            List containing index of encoder blocks to which forward hooks will be registered
        use_readout:str
            Appropriate readout operation,must be one of  {`add`,`ignore`,`project`}
        enable_attention_hooks:bool
            If True, forward hooks will be registered to attention blocks.
        start_index:int
            Parameter that handles readout operation, default value is 1.
        """

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
        """
        Makes a scratch module which is subclass of nn.Module

        Parameters
        ----------
        in_shape: list[int]
        out_shape:int
        groups: int
        expand:bool
        """
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
        """

        Parameters
        ----------
        features: int
            Number of features
        use_bn: bool
            Whether to use batch normalisation
        """
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
        Performs forward pass on backbone ViT model and fetches output from different encoder blocks with the help of hooks

        Parameters
        -----------
        x: torch.Tensor
            Input image tensor
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
        """
        Forward pass of DPTDepth

        Parameters
        ----------
        x:torch.Tensor
            Input image tensor
        """
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


class Slice(nn.Module):
    """Handles readout operation when `readout` parameter is `ignore`. Removes `cls_token` or  `readout_token` by index slicing"""

    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    """Handles readout operation when `readout` parameter is `add`. Removes `cls_token` or  `readout_token` from tensor and adds it to the rest of tensor"""

    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):

        readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    """Another class that handles readout operation. Used when `readout` parameter is `project`"""

    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Interpolate(nn.Module):
    """Interpolation module

    Parameters
    ----------
    scale_factor : float
        Scaling factor used in interpolation
    mode :str
        Interpolation mode
    align_corners: bool
        Whether to align corners in Interpolation operation
    """

    def __init__(self, scale_factor, mode, align_corners=False):

        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass"""

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module

    Parameters
    ----------
    features :int
        Number of features
    activation: nn.Module
        Activation module, default is nn.GELU
    bn: bool
        Whether to use batch normalisation
    """

    def __init__(self, features, activation=nn.GELU, bn=True):
        super().__init__()
        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """forward pass"""
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):

        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass"""
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    else:
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    return readout_oper
