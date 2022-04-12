import math
from torch import batch_norm, nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.structures import Instances, Boxes
from detectron2.layers import cat

from functools import reduce
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

# from nnunet.network_architecture.generic_UNet import Generic_UNet

from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.backbone import Backbone

INF = 100000000


class ConvDropoutNormNonlin(nn.Module):
    def __init__(
        self,
        input_channel,
        output_channel,
        kernel=3,
        stride=1,
        padding=1,
        norm=nn.BatchNorm3d,
        nonlin=nn.LeakyReLU,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            input_channel,
            output_channel,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        self.instnorm = norm(output_channel)
        self.lrelu = nonlin(inplace=True)

    def forward(self, x):
        return self.lrelu(self.instnorm(self.conv(x)))


class StackedConvLayers(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvDropoutNormNonlin(input_channel, output_channel, stride=stride),
            ConvDropoutNormNonlin(output_channel, output_channel),
        )

    def forward(self, x):
        return self.blocks(x)


class UNETD(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        stage_num = cfg.MODEL.UNETD.STAGE_NUM
        base_channel = cfg.MODEL.UNETD.BASE_CHANNELS
        max_channel = cfg.MODEL.UNETD.MAX_CHANNELS
        input_channel = input_shape.channels

        self.all_features_channels = [
            min(base_channel * 2 ** i, max_channel) for i in range(stage_num)
        ]
        self.per_strides = [1] + [2] * (stage_num - 1)
        prod = lambda x: reduce(operator.mul, x, 1)
        self.strides = [prod(self.per_strides[: i + 1]) for i in range(stage_num)]

        self.all_features_names = [f"res{i}" for i in range(stage_num)]
        self._out_feature_channels = dict(
            zip(self.all_features_names, self.all_features_channels)
        )
        self._out_feature_strides = dict(zip(self.all_features_names, self.strides))
        self._size_divisibility = 2 ** (stage_num - 1)
        self._out_features = cfg.MODEL.UNETD.OUT_FEATURES

        self.conv_blocks_context = nn.ModuleList()
        for i in range(stage_num):
            self.conv_blocks_context.append(
                StackedConvLayers(
                    input_channel, self.all_features_channels[i], self.per_strides[i]
                )
            )
            input_channel = self.all_features_channels[i]

        for module in [self.conv_blocks_context]:
            for l in module.children():
                if isinstance(l, (nn.Conv3d, nn.Conv2d)):
                    weight_init.c2_xavier_fill(l)
                # if isinstance(l, (nn.Conv3d, nn.Conv2d)):
                #     torch.nn.init.normal_(l.weight, std=0.01)
                #     torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        skips = []
        for d in range(len(self.conv_blocks_context)):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        out = dict(zip(self.all_features_names, skips))
        return {i: out[i] for i in self._out_features}


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet and FCOS to generate extra layers, P6 and P7 from
    C5 or P5 feature.
    """

    def __init__(self, in_channels, out_channels, in_features="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_features
        self.p6 = nn.Conv3d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv3d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class LastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_features="res5"):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_features
        self.p6 = nn.Conv3d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]


class FPN3D(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(
        self,
        bottom_up,
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum",
        lateral_layers=1,
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super().__init__()
        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            if lateral_layers == 1:
                lateral_conv = nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, bias=use_bias
                )
            else:
                # To expand size as nnunet
                lateral_conv = nn.Sequential(
                    nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm3d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=use_bias),
                )
            output_conv = nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            for m in lateral_conv.modules():
                if 'Conv' in m.__class__.__name__:
                    weight_init.c2_xavier_fill(m)

            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in strides
        }
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(
                    prev_features, scale_factor=2.0, mode="nearest"
                )
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                # print(self._out_features, self.top_block.in_feature, len(results))
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)
                ]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}


@BACKBONE_REGISTRY.register()
def build_fcos_unet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = UNETD(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS
    in_channels_top = out_channels
    if top_levels == 2:
        top_block = LastLevelP6P7(
            in_channels_top, out_channels, cfg.MODEL.FCOS.IN_FEATURES[-3]
        )
    if top_levels == 1:
        top_block = LastLevelP6(in_channels_top, out_channels, "p5")
    elif top_levels == 0:
        top_block = None
    backbone = FPN3D(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        lateral_layers=cfg.MODEL.FPN.LATERAL_LAYERS,
    )
    return backbone
