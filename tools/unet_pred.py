# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from functools import partial, reduce
import glob
from inspect import getframeinfo
import multiprocessing as mp
import os
import time
import cv2
import numpy as np
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
    # meta = [[r.scores, r.pred_boxes.tensor] if hasattr(r, 'pred_masks') else None for r in results]
    res =  torch.stack(res).float()
    ''' res: B, C, 128, 128, 128'''

    thres = bg_thres
    bkgrd = res.new_full(res.shape, thres)
    # print(res.unique(), res.shape)
    return torch.cat([bkgrd, res], dim=1)

def model_pred(x, model, bg_thres=0.5):
    '''
    x: b, c, s, h, w
    res: b, class, s, h, w'''
    pred_shape = (1,) + x.shape[-3:]
    x = prep_batch(x)
    y = model(x)
    res = extr_result(y, pred_shape, bg_thres)
    return res

from batchgenerators.augmentations.utils import pad_nd_image
from detectron2.structures import Instances, Boxes
def predict_3d(data, model, network, step_size, pad_border_mode='constant',pad_kwargs={'constant_values': 0}, use_gaussian=False, verbose=True, all_in_gpu=True, **kwargs):
    '''
    do_mirror unsupported
    data: 1, H, W, S'''
    assert use_gaussian==False
    assert all_in_gpu
    assert len(data.shape) == 4, "x must be (c, x, y, z)"

    patch_size = (128,128,128)
    data, slicer = pad_nd_image(data, patch_size, pad_border_mode, pad_kwargs, True, None)
    data_shape = data.shape  # still c, x, y, z
    if getattr(network,'cuthalf'):
        print('step cut half')
        steps = network._compute_steps_for_sliding_window_cuthalf(patch_size, data.shape[1:], step_size)
    else:
        steps = network._compute_steps_for_sliding_window(patch_size, data.shape[1:], step_size)
    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])
    if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)
    data = torch.from_numpy(data).cuda(network.get_device(), non_blocking=True)
        

    st = time.time()
    patches = []
    for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]
                    this_patch = prep_batch(data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z])
                    pred = model(this_patch)[0]['instances']
                    if len(pred):
                        new_pred = Instances((0,0))
                        keep_fields = ['pred_boxes', 'scores', 'pred_masks', 'pred_classes', 'pred_global_masks']
                        for k in keep_fields:
                            new_pred.set(k, pred.get(k))
                        pred = new_pred
                        pred_pos = data.new_tensor([lb_x, lb_y, lb_z, ub_x, ub_y, ub_z])
                        pred.pred_boxes.tensor = pred.pred_boxes.tensor + pred_pos[:3].repeat(2)
                        pred.pred_pos = pred_pos.repeat(len(pred),1)
                        patches.append(pred)
    print('avg time spent on patch {}'.format((time.time()-st)/num_tiles))

    # do nms  ~~voting~~
    print('before nms {} inst'.format(len(patches)))
    boxlists = Instances.cat(patches) if len(patches) else pred
    boxlists = model.proposal_generator.fcos_outputs.select_over_all_levels([boxlists], post_nms_topk=None)[0]
    # groups = []
    # boxlists.group = groups
    print('after nms {} inst'.format(len(boxlists)))

    if verbose: print("initializing result array (on GPU)")
    aggregated_results = torch.zeros([network.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=data.device)
    aggregated_results[0] = network.ffbg_thres    
    for i in range(len(boxlists)):
        box = boxlists[i]
        pp = box.pred_pos[0].int()
        sli = np.s_[pp[0]:pp[3], pp[1]:pp[4], pp[2]:pp[5]]
        aggregated_results[1][sli] = aggregated_results[1][sli] + box.pred_global_masks[0]
    # aggregated_results = (aggregated_results>0).float()

    # reverse padding
    slicer = tuple(
        [slice(0, aggregated_results.shape[i]) for i in
        range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
    aggregated_results = aggregated_results[slicer]
    
    if verbose:
        predicted_segmentation = (aggregated_results.argmax(dim=0)).detach().cpu().numpy()
        print('from neural_network, cls:', np.unique(predicted_segmentation, return_counts=True))
    if verbose: print("prediction done")
    return None, aggregated_results.detach().cpu().numpy()

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
        model.do_ds = False

        trainer, gen = get_generator(cfg, return_trainer=True)
        use_g = cfg.EVAL.USE_GAUSSIAN or False
        bg_t = cfg.EVAL.BG_THRES or 0.5
        trainer.network.ffbg_thres = bg_t
        trainer.network.cuthalf= cfg.EVAL.get('CUTHALF') or False
        if cfg.EVAL.PATCH_NMS:
            ppdrss = partial(predict_3d, model = model, network = trainer.network)
            trainer.predict_preprocessed_data_return_seg_and_softmax = ppdrss
        else:
            trainer.network.forward = partial(model_pred, model=model, bg_thres = bg_t)
            trainer.network.inference_apply_nonlin = lambda x:x

        # test specific case
        keys = [12] 
        # keys = ['case_{:05d}'.format(i) for i in keys] # for task kidney
        keys = ['train_{:02d}'.format(i) for i in keys] # for task liver
        # trainer.dataset_val = {k: trainer.dataset_val[k] for k in keys}
        st = time.time()
        ret = trainer.validate(save_softmax=False, do_mirroring=False, debug=False, validation_folder_name=cfg.EVAL.SAVE_DIR,
                            run_postprocessing_on_folds=False, use_gaussian=use_g, all_in_gpu=True,
                         overwrite=True,
                         step_size=cfg.EVAL.STEP_SIZE or 0.4)
        print(ret['mean']['1']['Dice'])
        print('time spent is {} min'.format((time.time()-st)/60))
    with open('tmp.yaml','w') as fi:
        fi.write(cfg.dump())
    print('finished')