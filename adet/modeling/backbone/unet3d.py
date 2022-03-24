import logging
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit

from adet.utils.comm import reduce_sum, reduce_mean, compute_ious
from adet.layers import ml_nms, IOULoss

from detectron2.structures import Instances, Boxes
from detectron2.layers import cat

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

from fvcore.nn import sigmoid_focal_loss_jit
from adet.utils.comm import reduce_mean

from adet.utils.gridding import GriddingReverse
from adet.utils.chamfer_distance import ChamferDistance

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
        in_channels = out_channels
    return mods

def compute_ctrness_targets_3d(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 3]]
    top_bottom = reg_targets[:, [1, 4]]
    near_far = reg_targets[:, [2, 5]]

    ctrness = (
        (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
        * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        * (near_far.min(dim=-1)[0] / near_far.max(dim=-1)[0])
    )
    return torch.sqrt(ctrness)

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


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

        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.radius = cfg.MODEL.FCOS.POS_RADIUS
        self.num_classes = cfg.MODEL.MASKHEAD.NUM_CLASSES
        self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
        self.loc_loss_func = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)
        self.box_quality = cfg.MODEL.FCOS.BOX_QUALITY
        self.loss_normalizer_cls = cfg.MODEL.FCOS.LOSS_NORMALIZER_CLS
        self.loss_weight_cls = cfg.MODEL.FCOS.LOSS_WEIGHT_CLS

        self.in_features = sorted(input_shape.keys())
        # Need to modify
        self.stride = 2**5

        self.cls_tower = nn.Sequential(*make_conv(256, 64, repeated=4))
        self.cls_logits = nn.Conv3d(64, cfg.MODEL.MASKHEAD.NUM_CLASSES, 3, 1, 1)
        self.cls_pixpred = nn.Conv3d(64, cfg.MODEL.MASKHEAD.NUM_CLASSES, 3, 1, 1)

        self.bbox_tower = nn.Sequential(*make_conv(256, 64, repeated=4))
        self.ctrness = nn.Conv3d(64, 1, 3, 1, 1)
        self.reg = nn.Conv3d(64, 6, 1, 1)

        self.shape_enabled = True
        self.chamfer_dist = ChamferDistance()
        self.grid_rev = GriddingReverse(64)
        self.label_pc = []
        template_path = '../adet/utils/gridding/template_pc'
        with open(template_path, 'r') as f:
            for line in f:
                infos = [float(x) for x in line.split(' ')]
                self.label_pc.append(infos)
        self.label_pc = torch.tensor(self.label_pc).cuda()


        for modules in [
            self.cls_tower,
            self.cls_logits,
            self.cls_pixpred,
            self.bbox_tower,
            self.ctrness,
            self.reg,
        ]:
            for l in modules.modules():
                if isinstance(l, (nn.Conv3d, nn.Conv2d)):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.register_buffer("_iter", torch.zeros([1]))

    # temp 3d full size location
    def compute_locations_3d(self, feature):
        s, h, w = feature.shape[-3:]
        s = torch.arange(s)
        h = torch.arange(h)
        w = torch.arange(w)
        return (
            torch.stack(torch.meshgrid(s, h, w), dim=3).to(feature.device)
        )

    def _get_ground_truth(self, locations, gt_instances):
        training_targets = {}
        labels = []
        reg_targets = []
        s,h,w = locations.shape[:3]
        locations = locations.view(-1,3)
        xs, ys, zs = locations[:, 0], locations[:, 1], locations[:, 2]

        for im_i in range(len(gt_instances)):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes.long()

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            n = zs[:, None] - bboxes[:, 2][None]
            r = bboxes[:, 3][None] - xs[:, None]
            b = bboxes[:, 4][None] - ys[:, None]
            f = bboxes[:, 5][None] - zs[:, None]
            reg_targets_per_im = torch.stack([l, t, n, r, b, f], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(
                dim=1
            )

            reg_targets_per_im = reg_targets_per_im[
                range(len(locations)), locations_to_gt_inds
            ]

            # normalize regression
            reg_targets_per_im = reg_targets_per_im / (self.stride*1.5)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        training_targets.update(
            {
                "labels": labels,
                "reg_targets": reg_targets,
            }
        )
        return training_targets

    def _proposal_losses(
        self,
        logits_pred,
        reg_pred,
        ctrness_pred,
        locations,
        gt_instances,
    ):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets = self._get_ground_truth(locations, gt_instances)

        # Collect all logits and regression predictions over feature maps
        # and images to gt_instances at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.

        instances = Instances((0, 0))
        instances.labels = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1)
                for x in training_targets["labels"]
            ],
            dim=0,
        )
        instances.reg_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 6)
                for x in training_targets["reg_targets"]
            ],
            dim=0,
        )

        instances.logits_pred = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in logits_pred
            ],
            dim=0,
        )
        instances.reg_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 6)
                for x in reg_pred
            ],
            dim=0,
        )
        instances.ctrness_pred = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.permute(0, 2, 3, 1).reshape(-1)
                for x in ctrness_pred
            ],
            dim=0,
        )

        return self.fcos_losses(instances)

    def fcos_losses(self, instances):
        losses = {}

        # 1. compute the cls loss
        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes

        labels = instances.labels.flatten()

        pos_inds = torch.nonzero(labels != 0).squeeze(1)

        num_pos_local = torch.ones_like(pos_inds).sum()
        num_pos_avg = max(reduce_mean(num_pos_local).item(), 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(instances.logits_pred)
        # only one class
        class_target[pos_inds, 0] = 1

        class_loss = sigmoid_focal_loss_jit(
            instances.logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        if self.loss_normalizer_cls == "moving_fg":
            self.moving_num_fg = (
                self.moving_num_fg_momentum * self.moving_num_fg
                + (1 - self.moving_num_fg_momentum) * num_pos_avg
            )
            class_loss = class_loss / self.moving_num_fg
        elif self.loss_normalizer_cls == "fg":
            class_loss = class_loss / num_pos_avg

        losses["loss_fcos_cls"] = class_loss * self.loss_weight_cls

        # 2. compute the box regression and quality loss
        instances = instances[pos_inds]
        instances.pos_inds = pos_inds

        ious, gious = compute_ious_3d(instances.reg_pred, instances.reg_targets)

        if self.box_quality == "ctrness":
            ctrness_targets = compute_ctrness_targets_3d(instances.reg_targets)
            instances.gt_ctrs = ctrness_targets

            ctrness_targets_sum = ctrness_targets.sum()
            loss_denorm = max(reduce_mean(ctrness_targets_sum).item(), 1e-6)
            # rangef = lambda x: (x.min(), x.max())
            # print(loss_denorm, rangef(ctrness_targets))
            # print(rangef(ious), rangef(gious))

            reg_loss = self.loc_loss_func(ious, gious, ctrness_targets) / loss_denorm
            losses["loss_fcos_loc"] = reg_loss

            ctrness_loss = (
                F.binary_cross_entropy_with_logits(
                    instances.ctrness_pred, ctrness_targets, reduction="sum"
                )
                / num_pos_avg
            )
            losses["loss_fcos_ctr"] = ctrness_loss

        return losses

    def forward(self, images, features, gt_instances=None):
        if self.training:
            seg_in = features[self.in_features[0]]
            cls_in = self.cls_tower(seg_in)
            pixpred = self.cls_pixpred(cls_in)
            seg_output = self.cls_logits(cls_in)

            reg_in = self.bbox_tower(seg_in)
            ctrness = self.ctrness(reg_in)
            reg_box = self.reg(reg_in)

            losses = {}
            locations = self.compute_locations_3d(seg_in)
            proposal_loss = self._proposal_losses(
                seg_output, reg_box, ctrness, locations, gt_instances
            )
            losses.update(proposal_loss)

            self._iter += 1
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])[
                :, None
            ]
            # gt_bitmasks[0,0,60]
            mask_logits = pixpred
            mask_scores = mask_logits.sigmoid()

            loss_pc = 0
            if self.shape_enabled:
                pred = pixpred.unsqueeze(-1)
                pred = torch.cat([pred, 1 - pred], dim=-1)
                pred = gumbel_softmax(pred, hard=True)
                pred_pc = self.grid_rev(pred[:, :, :, :, 0].contiguous())
                loss_pc = self.chamfer_dist(pred_pc, self.label_pc)

            if self.boxinst_enabled:
                # box-supervised BoxInst losses
                image_color_similarity = torch.cat(
                    [x.image_color_similarity for x in gt_instances]
                )
                # print(mask_scores.shape, gt_bitmasks.shape)

                loss_prj_term = compute_project_term_3d(mask_scores, gt_bitmasks)
                pairwise_losses = compute_pairwise_term_3d(
                    mask_logits, self.pairwise_size, self.pairwise_dilation
                )
                # print(pairwise_losses.shape)

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
                        "loss_pc": loss_pc,
                    }
                )

                return mask_scores, losses
        else:
            seg_in = list(features.values())[0]
            cls_in = self.cls_tower(seg_in)
            pixpred = self.cls_pixpred(cls_in)
            # for i, r in enumerate([result]):
            #     results[i] = r.sigmoid
            #     results[i] = (results[i] > 0.5).int()
            result = (pixpred.sigmoid() > 0.5).int()
            extras = {}
            return result, extras


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

    from adet.modeling.condinst.condinst3d import unfold_wo_center_3d

    log_fg_prob_unfold = unfold_wo_center_3d(
        log_fg_prob, kernel_size=pairwise_size, dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center_3d(
        log_bg_prob, kernel_size=pairwise_size, dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = (
        torch.log(
            torch.exp(log_same_fg_prob - max_) + torch.exp(log_same_bg_prob - max_)
        )
        + max_
    )

    # loss = -log(prob)
    return -log_same_prob[:, 0]

def compute_ious_3d(pred, target):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_n = pred[:, 2]
    pred_right = pred[:, 3]
    pred_bottom = pred[:, 4]
    pred_f = pred[:, 5]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_n = target[:, 2]
    target_right = target[:, 3]
    target_bottom = target[:, 4]
    target_f = target[:, 5]

    target_aera = (
        (target_left + target_right)
        * (target_top + target_bottom)
        * (target_n + target_f)
    )
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom) * (pred_n + pred_f)

    w_intersect = torch.min(pred_left, target_left) + torch.min(
        pred_right, target_right
    )
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
        pred_top, target_top
    )
    s_intersect = torch.min(pred_n, target_n) + torch.min(pred_f, target_f)

    g_w_intersect = torch.max(pred_left, target_left) + torch.max(
        pred_right, target_right
    )
    g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
        pred_top, target_top
    )
    g_s_intersect = torch.max(pred_n, target_n) + torch.max(pred_f, target_f)
    ac_uion = g_w_intersect * g_h_intersect * g_s_intersect

    # print("pred :{}".format(pred))
    # print("target :{}".format(target))

    area_intersect = w_intersect * h_intersect * s_intersect
    area_union = target_aera + pred_area - area_intersect

    # print("target_area: {}".format(target_aera))
    # sht = locals()
    # printde = lambda x: print("{}: {}".format(x, sht[x]))
    # printde('pred_area')
    # printde('area_intersect')

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion

    return ious, gious
