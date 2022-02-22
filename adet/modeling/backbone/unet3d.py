from functools import reduce
import operator
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# from nnunet.network_architecture.generic_UNet import Generic_UNet

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.modeling.condinst.dynamic_mask_head import (
    compute_pairwise_term,
    compute_project_term,
    dice_coefficient,
)


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


# @META_ARCH_REGISTRY.register()
class unet3d(Backbone):
    def __init__(
        self,
        cfg,
        input_shape,
        stage_num=5,
        base_channel=16,
        max_channel=320,
        out_channels=256,
    ):
        super().__init__()
        input_channel = input_shape.channels
        self.all_features_channels = [
            min(base_channel * 2 ** i, max_channel) for i in range(stage_num + 1)
        ]
        self.per_strides = [1] + [2] * (
            stage_num
        )  # [min(2**i, 2) for i in range(stage_num)]
        prod = lambda x: reduce(operator.mul, x, 1)
        self.strides = [prod(self.per_strides[:i]) for i in range(stage_num + 1)]

        self.all_heads_names = [f"p{i+1}" for i in range(stage_num)]
        self._out_feature_channels = dict(
            zip(self.all_heads_names, [out_channels] * len(self.all_heads_names))
        )
        self._out_feature_strides = dict(zip(self.all_heads_names, self.strides[:-1]))
        self._size_divisibility = None
        self._out_features = cfg.MODEL.UNET3D.OUT_FEATURES

        self.conv_blocks_context = nn.ModuleList()
        for i in range(stage_num + 1):
            self.conv_blocks_context.append(
                StackedConvLayers(
                    input_channel, self.all_features_channels[i], self.per_strides[i]
                )
            )
            input_channel = self.all_features_channels[i]

        self.conv_blocks_localization = nn.ModuleList()
        for i in range(stage_num):
            input_channel = self.all_features_channels[-i - 2]
            self.conv_blocks_localization.append(
                nn.Sequential(
                    StackedConvLayers(input_channel * 2, input_channel),
                    StackedConvLayers(input_channel, input_channel),
                )
            )

        self.tu = nn.ModuleList()
        for i in range(stage_num):
            self.tu.append(
                nn.ConvTranspose3d(
                    self.all_features_channels[-i - 1],
                    self.all_features_channels[-i - 2],
                    self.per_strides[-i - 1],
                    self.per_strides[-i - 1],
                    bias=False,
                )
            )

        self.output_head = nn.ModuleList()
        for i in self.all_features_channels[:-1][::-1]:
            self.output_head.append(nn.Conv3d(i, out_channels, 3, 1, 1, bias=False))

    def forward(self, x):
        skips = []
        for d in range(len(self.conv_blocks_context)):
            x = self.conv_blocks_context[d](x)
            if d < len(self.conv_blocks_context) - 1:
                skips.append(x)

        output = []
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            # print(x.shape, skips[-u - 1].shape)
            x = torch.cat([x, skips[-u - 1]], dim=1)
            x = self.conv_blocks_localization[u](x)
            out = self.output_head[u](x)
            # out = x
            output.append(out)
        output = output[::-1]

        out_dict = {n: i for n, i in zip(self._out_features, output)}
        return {n: out_dict[n] for n in self._out_features}


@BACKBONE_REGISTRY.register()
def build_unet3d(cfg, input_shape):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    base_channel = 32
    backbone = unet3d(cfg, input_shape, base_channel=base_channel)
    return backbone


def make_conv(in_channels, out_channels, repeated=3):
    mods = []
    for _ in range(repeated):
        mods += [
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        ]
    return mods


@PROPOSAL_GENERATOR_REGISTRY.register()
class MaskHead(nn.Module):
    # input_shape: {p_i: channel, stride}
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS

        self.in_features = sorted(input_shape.keys())
        # self.cls_tower = nn.ModuleList()
        self.cls_logits = nn.ModuleList()
        for _ in input_shape:
            # self.cls_tower.append(nn.Sequential(*make_conv(256, 256, repeated=1)))
            self.cls_logits.append(
                nn.Conv3d(256, cfg.MODEL.MASKHEAD.NUM_CLASSES, 3, 1, 1)
            )

        for modules in [
            # self.cls_tower,
            self.cls_logits,
        ]:
            for l in modules.modules():
                if isinstance(l, (nn.Conv3d, nn.Conv2d)):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.register_buffer("_iter", torch.zeros([1]))

    def forward(self, images, features, gt_instances=None):
        if self.training:
            seg_in = features[self.in_features[0]]
            seg_output = self.cls_logits[0]((seg_in))
            # print(seg_output.shape)
            losses = {}
            if gt_instances[0].has("logits_pred"):
                proposal_loss = self._proposal_loss(seg_output, gt_instances)
                losses.update(proposal_loss)

            self._iter += 1
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])[:,None]

            mask_logits = seg_output
            mask_scores = mask_logits.sigmoid()

            if self.boxinst_enabled:
                # box-supervised BoxInst losses
                image_color_similarity = torch.cat(
                    [x.image_color_similarity for x in gt_instances]
                )
                print(mask_scores.shape, gt_bitmasks.shape)

                loss_prj_term = compute_project_term_3d(mask_scores, gt_bitmasks)
                pairwise_losses = compute_pairwise_term_3d(
                    mask_logits, self.pairwise_size, self.pairwise_dilation
                )
                print(pairwise_losses.shape)

                weights = (
                    image_color_similarity >= self.pairwise_color_thresh
                ).float() * gt_bitmasks.float()
                loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(
                    min=1.0
                )

                warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                loss_pairwise = loss_pairwise * warmup_factor

                losses.update(
                    {
                        "loss_prj": loss_prj_term,
                        "loss_pairwise": loss_pairwise,
                    }
                )
                print(losses)
                return losses
        else:
            results = OrderedDict(features).values()
            for i, r in enumerate(results):
                results[i] = self.inference_apply_nonlin(r)
                results[i] = (results[i] > 0.5).int()
            extras = {}
            return results[-1], extras

    def _proposal_loss(self, seg_output, gt_instances):
        gt_mask = gt_instances.logits_pred
        return F.binary_cross_entropy_with_logits(seg_output, gt_mask)


def compute_project_term_3d(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0], gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0], gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    mask_losses_z = dice_coefficient(
        mask_scores.max(dim=4, keepdim=True)[0], gt_bitmasks.max(dim=4, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y + mask_losses_z).mean()


def compute_pairwise_term_3d(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 5

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center_3d
    log_fg_prob_unfold = unfold_wo_center_3d(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center_3d(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]
