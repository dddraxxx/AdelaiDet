# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from functools import partial, reduce
import glob
from inspect import getframeinfo
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
from adet.utils.nnunet_generator import nnUNet_loader, get_generator, args as ar, remove_all_but_the_two_largest_conn_comp

from demo.predictor3d import VisualizationDemo
from adet.config import get_cfg
from adet.utils.volume_utils import read_niigz
from detectron2.config import CfgNode

from adet.utils.dataset_3d import Volumes, get_dataset, read_volume, save_volume
from adet.utils.visualize_niigz import (
    PyTMinMaxScalerVectorized,
    visulize_3d,
    draw_3d_box_on_vol,
)

from gpu_stat import get_free_gpu
import os
os.environ['CUDA_VISIBLE_DEVICES']= str(get_free_gpu())

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
    '''
    print image, depth, height, width'''
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

def prep_batch(x):
    imgs = x
    input = []
    for i in imgs:
        depth, height, width = i.shape[-3:]
        input.append(dict(image=i, depth=depth, height=height, width=width))
    return input

def extr_result(y, shape, bg_thres=0.5):
    # print(y[0].keys(), y[0]['instances']._fields)
    results = [r["instances"] for r in y]
    res = [(r.pred_masks.sum(dim=0, keepdim=True)>0).int() if hasattr(r, 'pred_masks') else r.top_feat.new_zeros(shape) for r in results]
    meta = [[r.scores, r.pred_boxes.tensor] if hasattr(r, 'pred_masks') else None for r in results]
    res =  torch.stack(res).float()
    ''' res: B, C, 128, 128, 128'''

    thres = bg_thres
    bkgrd = res.new_full(res.shape, thres)
    # print(res.unique(), res.shape)
    return torch.cat([bkgrd, res], dim=1), meta

def model_pred(x, model, bg_thres=0.5):
    '''
    x: b, c, s, h, w
    res: b, class, s, h, w'''
    pred_shape = (1,) + x.shape[-3:]
    x = prep_batch(x)
    y = model(x)
    res, meta = extr_result(y, pred_shape, bg_thres)
    return res

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    model = demo.predictor.model
    model.eval()

    with torch.no_grad():
        trainer, gen = get_generator(cfg, return_trainer=True)
        bg_t = cfg.EVAL.BG_THRES if (cfg.EVAL.get('BG_THRES') is not None) else 0.5
        use_g = cfg.EVAL.USE_GAUSSIAN if cfg.EVAL.get('USE_GAUSSIAN') is not None else False
        trainer.network.forward = partial(model_pred, model=model, bg_thres = bg_t)
        trainer.network.inference_apply_nonlin = lambda x:x
        trainer.network.cuthalf=True
        trainer.network.keep_complete = False
        model.do_ds = False
        # test specific case
        keys = [233, 113] # 273 lower inf_test to 0.65
        keys = ['case_{:05d}'.format(i) for i in keys]
        # trainer.dataset_val = {k: trainer.dataset_val[k] for k in keys}
        st = time.time()
        ret = trainer.validate(save_softmax=False, do_mirroring=False, debug=False, validation_folder_name=cfg.EVAL.SAVE_DIR,
                            run_postprocessing_on_folds=False, use_gaussian=use_g, all_in_gpu=True,
                         overwrite=True,
                         step_size=0.4)
        print(ret['mean']['1']['Dice'])
        print('time spent is {} min'.format((time.time()-st)/60))
    with open('tmp.yaml','w') as fi:
        fi.write(cfg.dump())
    print('finished')