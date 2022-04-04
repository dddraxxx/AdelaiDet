# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from functools import reduce
import glob
import multiprocessing as mp
import os
import time
import cv2
import torch
import tqdm
from torchvision.utils import draw_segmentation_masks
from torch.utils.data import DataLoader

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from adet.utils.comm import aligned_bilinear3d

from predictor3d import VisualizationDemo
from adet.config import get_cfg
from adet.utils.volume_utils import read_niigz
from detectron2.config import CfgNode

from adet.utils.dataset_3d import Volumes, get_dataset, read_volume, save_volume
from adet.utils.visualize_niigz import (
    PyTMinMaxScalerVectorized,
    visulize_3d,
    draw_3d_box_on_vol,
)

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg_3d(args):
    cfg = CfgNode()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input", nargs="+", help="A list of space separated input images"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def pred_batch(batch, model):
    imgs = [i["image"] for i in batch]
    gt = [i["instances"].gt_masks.cpu() for i in batch]
    idx = [i["index"] for i in batch]
    input = []
    for i in imgs:
        depth, height, width = i.shape[-3:]
        input.append(dict(image=i, depth=depth, height=height, width=width))
    with torch.no_grad():
        results = model(input)
    results = [r["instances"].to("cpu") for r in results]
    return results, gt, idx


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    model = demo.predictor.model

    # now pred cases 240-300
    from detectron2.data.build import trivial_batch_collator
    pred_ds = get_dataset(60)
    pred_ds.data = list(range(240,300))
    pred_ds.data.remove(296)
    pred_dl = DataLoader(pred_ds, 10, collate_fn= trivial_batch_collator, shuffle=False, pin_memory=True,   num_workers=12, drop_last=False)
    res = []
    for batch in iter(pred_dl):
        print([i['index'] for i in batch])
        re = pred_batch(batch, demo.predictor.model)
        re = list(zip(*re))
        print(len(re))
        res.extend(re)
    from pathlib import Path as pa
    dest = pa('pred_fu1')
    dest.mkdir(exist_ok=True)
    for r in res:
        d = dest / '{:05d}.npy'.format(r[-1])
        print(f'save to {d}')
        torch.save(r, d)
    print('finished')