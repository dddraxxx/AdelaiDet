from typing import List
from detectron2.layers import batched_nms3d, batched_nms
import torch

from adet.utils.dataset_3d import Boxes3D


def ml_nms(
    boxlist, nms_thresh, max_proposals=-1, score_field="scores", label_field="labels"
):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Args:
        boxlist (detectron2.structures.Boxes):
        nms_thresh (float):
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str):
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist


def ml_nms3d(
    boxlist, nms_thresh, max_proposals=-1, score_field="scores", label_field="labels"
):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Args:
        boxlist (detectron2.structures.Boxes):
        nms_thresh (float):
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str):
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = monai_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    # print(keep)
    # boxlist = boxlist[keep]
    # print(boxlist)
    return boxlist


def monai_nms(boxes, scores, idxs, nms_thresh):
    assert boxes.shape[-1] == 6
    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.jit.annotate(List[int], torch.unique(idxs).cpu().tolist()):
        mask = (idxs == id).nonzero().view(-1)
        keep = nms3d(boxes[mask], scores[mask], nms_thresh)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def nms3d(boxes, scores, thresh):
    assert len(boxes) == len(scores)
    bx = Boxes3D(boxes)
    l = torch.arange(len(boxes))
    le = len(boxes)
    # print(len(bx))
    r, c = torch.meshgrid(l, l, indexing="ij")
    tables = bx.iou(r.reshape(-1), c.reshape(-1)).view(le, le)
    tables = tables < thresh
    tables[(*torch.triu_indices(le, le),)] = True
    res = tables.all(1)
    return res
