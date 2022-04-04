#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from adet.utils.dataset_3d import *

args = argparse.Namespace(continue_training=False, deterministic=False, disable_next_stage_pred=False, disable_postprocessing_on_folds=False, disable_saving=False, find_lr=False, fold='0', fp32=True, network='3d_lowres', network_trainer='nnUNetTrainerV2', npz=False, p='nnUNetPlansv2.1', pretrained_weights=None, task='135', use_compressed_data=False, val_disable_overwrite=True, val_folder='validation_raw', valbest=False, validation_only=False)

def get_generator(cfg=None, mode='train', return_trainer=False):
    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder
    # interp_order = args.interp_order
    # interp_order_z = args.interp_order_z
    # force_separate_z = args.force_separate_z

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    # if force_separate_z == "None":
    #     force_separate_z = None
    # elif force_separate_z == "False":
    #     force_separate_z = False
    # elif force_separate_z == "True":
    #     force_separate_z = True
    # else:
    #     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)

    trainer.uinst_dct = {}
    dct = trainer.uinst_dct
    if cfg:
        trainer.uinst_dct.update({'batch_size': cfg.SOLVER.IMS_PER_BATCH})
        trainer.uinst_dct.update(num_classes=cfg.MODEL.FCOS.NUM_CLASSES + 1)
    trainer.uinst_dct.update({'deep_supervision_scales': None})
    
    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only)

    assert mode in ['train', 'val']    
    if mode=='train':
        gen = trainer.tr_gen
    else:
        gen = trainer.val_gen
    # print(next(gen).keys())
    # print(gen.next().keys())
    # print(next(gen)['data'].shape)
    # print((next(gen)['target']).shape)
    # import numpy as np
    # dl_tr = trainer.dl_tr
    # dl_val = trainer.dl_val
    # seg=dl_val.generate_train_batch()['seg']
    # print(seg.shape, np.unique(seg))
    if return_trainer:
        return trainer, gen
    return gen

def remove_all_but_the_two_largest_conn_comp(img_npy, thres=1e3):
    labelmap, num_labels = label((img_npy > 0).astype(int))

    if num_labels > 1:
        label_sizes = []
        for i in range(1, num_labels + 1):
            label_sizes.append(np.sum(labelmap == i))
        argsrt = np.argsort(label_sizes)[::-1] # two largest are now argsrt[0] and argsrt[1]

        keep_mask=[]
        for idx in argsrt[:2]:
            print('this kidney has size {}'.format(label_sizes[idx]))
            if label_sizes[idx]>thres:
                keep_mask.append(labelmap == idx+1)
        
        img_npy = np.stack(keep_mask).astype(int) if len(keep_mask) else np.zeros_like(img_npy[None])
    else:
        img_npy = img_npy[None]
    return img_npy
        
        

class nnUNet_loader:
    def __init__(self, cfg):
        self.gen = get_generator(cfg)
        self.seg = cfg.MODEL.CONDINST.ONLY_SEG
        _  = self.gen.next()
    
    @staticmethod
    def create_instance_by(data, seg, seg_or_inst):
        '''data: 1, 128, 128, 128
        seg: 1, 128, 128, 128'''
        seg = (seg==1).numpy()
        if not seg_or_inst:
            filtered_seg = remove_all_but_the_two_largest_conn_comp(seg[0], thres=5e2)
        else: filtered_seg = seg
        filtered_seg = torch.from_numpy(filtered_seg).float()
        # print(filtered_seg.shape)
        labels = T.BoundingRect()(filtered_seg)[:, [0, 2, 4, 1, 3, 5]]
        # print(labels.shape)
        print('total {} kidneys for this instance'.format(len(labels)))
        print('max edge for label: {}'.format((labels[:, 3:]-labels[:, :3]).max()/2))

        gt_instance = Instances((0, 0))
        gt_boxes = Boxes3D(labels)
        gt_instance.gt_boxes = gt_boxes
        gt_instance.gt_classes = torch.zeros(len(labels)).long()
        gt_instance.gt_masks = filtered_seg
        return {
            "image": data,
            "instances": gt_instance,
            "height": 128,
        }

    def __iter__(self):
        while True:
            dct = self.gen.next()
            data = dct['data']
            seg = dct['target']
            instances = [[dct[i][j] for i in ['data', 'target']] for j in range(len(dct['data']))]
            instances = [self.create_instance_by(*i, seg_or_inst=self.seg) for i in instances]
            yield instances


if __name__ == "__main__":
    get_generator()
