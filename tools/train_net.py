# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

from functools import wraps
import logging
import os
from collections import OrderedDict
from pprint import pprint
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from detectron2.structures import Instances, Boxes
from detectron2.data.build import trivial_batch_collator

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

from torch.utils.data import Dataset, DataLoader

import monai.transforms as T

from adet.utils.dataset_3d import Copier, get_dataset, Boxes3D
from adet.utils.dataset_2d import get_dataset2d
from adet.utils.nnunet_generator import get_generator, nnUNet_loader
from ValidateHook import build_val_hook

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class random3D(Dataset):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.data = [torch.rand(1, 250, 400, 400) for _ in range(3)]
        self.crop = T.RandSpatialCrop(
            (64 * 2, 64 * 2, 64 * 2), random_center=False, random_size=False
        )
        self.normalizer = lambda x: (x - x.mean(dim=[1, 2, 3], keepdim=True)) / x.std(
            dim=[1, 2, 3], keepdim=True
        )

    def __getitem__(self, index):
        index = index % len(self.data)
        x = self.data[index]
        gt_instance = Instances((0, 0))
        gt_boxes = Boxes3D(torch.tensor([40, 40, 40, 60, 60, 60])[None])
        gt_instance.gt_boxes = gt_boxes
        gt_instance.gt_classes = torch.tensor([0])
        size = dict(height=128, width=64, depth=256)
        x = self.normalizer(self.crop(x))
        return {"image": x, "instances": gt_instance, **size}

    def __len__(self):
        return self.length


class random2D(Dataset):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.data = [torch.rand(3, 100, 100) for _ in range(3)]

    def __getitem__(self, index):
        index = index % len(self.data)
        x = self.data[index]
        gt_instance = Instances((0, 0))
        gt_boxes = Boxes(
            torch.tensor(([40, 60, 60, 80], [10, 10, 40, 50], [70, 70, 90, 85]))
        )
        gt_instance.gt_boxes = gt_boxes
        gt_instance.gt_classes = torch.tensor([2, 1, 2])
        size = dict(height=128, width=128)
        return {"image": x, "instances": gt_instance, **size}

    def __len__(self):
        return self.length


from detectron2.engine import HookBase


class Freezer(HookBase):
    """backbone freezer"""

    def __init__(self, cfg):
        self.freeze_iter = cfg.MODEL.BACKBONE.get("FREEZE_TILL")
        self.freeze_name = "bottom_up"

    def before_step(self):
        if self.trainer.iter + 1 <= self.freeze_iter:
            for n, m in self.trainer.model.named_modules():
                if self.freeze_name in n:
                    # print('freeze {}'.format(n))
                    for a in m.parameters():
                        a.require_grad = True
                    # m.eval()
        if self.trainer.iter + 1 > self.freeze_iter:
            # self.trainer.model.train()
            for n, m in self.trainer.model.named_modules():
                for a in m.parameters():
                    a.require_grad = True

from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params
from build_optimizer import get_optimizer_params
class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
            overrides={"bottom_up":{"lr": cfg.SOLVER.get("LR_RATIO_BOTTOM_UP", 1) * cfg.SOLVER.BASE_LR}}
        )
        return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    def add_hook(self, hook):
        self.register_hooks([hook])
        self._hooks = self._hooks[:-2] + self._hooks[-2:][::-1]

    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(
                    self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD
                )
        if self.cfg.get("VAL") and self.cfg.VAL.ENABLED:
            self.val_hook = build_val_hook(self.cfg)
            self.add_hook(self.val_hook)

        if self.cfg.MODEL.BACKBONE.get('PRETRAIN', False) and self.cfg.MODEL.BACKBONE.get("FREEZE_TILL", 0) > 0:
            self.freeze_hook = Freezer(self.cfg)
            self.add_hook(self.freeze_hook)
        return ret

    def resume_or_load(self, resume=True, show_pretrained=False):
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if not resume and show_pretrained:
            import torchvision.models as models

            r3d_18 = models.video.r3d_18(pretrained=True)
            x = torch.rand(1, 3, 16, 16, 16)
            x = r3d_18(x)
            print(r3d_18, file=open("3dres.js", "w"))
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                print("iter: {}".format(self.iter))
                self.before_step()
                self.run_step()
                self.after_step()
                torch.cuda.empty_cache()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if cfg.MODEL.FCPOSE_ON:
            mapper = FCPoseDatasetMapper(cfg, True)
        else:
            mapper = DatasetMapperWithBasis(cfg, True)
        # return build_detection_train_loader(cfg, mapper=mapper)
        # return DataLoader(random3D(cfg.SOLVER.MAX_ITER), 1, collate_fn=lambda x: x, shuffle=True)
        total_len = (
            cfg.SOLVER.MAX_ITER * cfg.SOLVER.IMS_PER_BATCH * comm.get_world_size()
        )
        if cfg.DATALOADER.get("TYPE") == "nnunet":
            return nnUNet_loader(cfg)
        if (
            "3d" not in cfg.MODEL.META_ARCHITECTURE.lower()
            and len(cfg.MODEL.PIXEL_MEAN) == 3
        ):
            return DataLoader(
                get_dataset2d(total_len),
                cfg.SOLVER.IMS_PER_BATCH,
                collate_fn=lambda x: x,
                shuffle=True,
                pin_memory=True,
                num_workers=cfg.SOLVER.IMS_PER_BATCH + 8,
            )
        return DataLoader(
            get_dataset(total_len),
            cfg.SOLVER.IMS_PER_BATCH,
            collate_fn=trivial_batch_collator,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.SOLVER.IMS_PER_BATCH + 8,
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)  # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(
        resume=args.resume, show_pretrained=cfg.MODEL.get("PRETRAIN", False)
    )
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    # check model size
    # model = get_generator(return_trainer=True)[0].network
    model = trainer._trainer.model
    print(model, file=open("3d_network.js", "w"))
    total_params = sum(p.nelement() * p.element_size() for p in model.parameters())
    total_buff = sum(b.nelement() * b.element_size() for b in model.buffers())
    print(
        "model buff size: {:.3f}MB, model param size:{:.3f}MB".format(
            total_buff / 1024 ** 2, total_params / 1024 ** 2
        ),
        file=open("3d_network.js", "a"),
    )

    def modelsize_dict(model):
        dct = {}
        dct.update(
            size=sum(p.nelement() * p.element_size() for p in model.parameters())
        )
        dct.update(dict((n, modelsize_dict(m)) for n, m in model.named_children()))
        dct = {model._get_name(): dct}
        return dct

    msize = modelsize_dict(model)
    pprint(
        msize,
        open("3d_network.js", "a"),
    )

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
