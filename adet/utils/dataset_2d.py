from functools import reduce
import numpy as np

from pathlib import Path as pa

import monai
import monai.transforms as T
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from itertools import product
from scipy.ndimage import find_objects, label

from detectron2.structures import Instances, Boxes

from torchvision.utils import save_image, draw_bounding_boxes

from adet.utils.dataset_3d import *


def get_dataset2d(length):
    return Slices(length)


class Slices(Volumes):
    def __init__(self, length):
        super().__init__(length)
        self.length = length
        # 10 samples total * 6 slices per sample
        # case236, case297 has no label
        self.data = list(range(0,80))
        self.prep_data=list(range(236, 300))
        self.sl_len = 6
        self.total_len = 10
        self.crop_size = None
        # self.crop = T.RandSpatialCrop(
        #     (128,128,128), random_center=False, random_size=False
        # )
        self.normalizer = lambda x: (x - x.mean(dim=[-1, -2, -3], keepdim=True)) / x.std(
            dim=[-1, -2, -3], keepdim=True
        )
        self.data_path_nontight_ntr = "/mnt/sdc/kits21/data_2d_notranspose/{:05d}.pt"
        self.data_path_nontight = "/mnt/sdc/kits21/data_2d/{:05d}.pt"
        self.data_path_tight = '/mnt/sdc/kits21/data_2d_tightbox/{:05d}.pt'
        self.data_path_tt_notr = "/mnt/sdc/kits21/data_2d_tightbox_notranspose/{:05d}.pt"
        self.data_path = self.data_path_nontight

    def _prepare_data(self, num_slices=None, tight_box=False, transpose=True, save_path="/mnt/sdc/kits21/data_2d"):
        """
        data, gt: 1,N,H,W
        label: 1, N, 4"""
        for i in self.prep_data:
            x, label = self.read_data(i, read_gt=True, transpose=transpose)
            x[0] = self.normalizer(x[0])
            s = x.size(1)

            # print(label.unique(), label.shape)
            # read_data has read all the label
            if len(label):
                lb = label[:1]
                lb = lb.int()
                st = lb[0, 0]
                ed = lb[0, 3]
                
                # tight box
                if tight_box:
                    lb2 = torch.zeros(s, 4)
                    lb = lb[0]
                    gt1 = x[1]
                    tmp = torch.zeros_like(gt1)
                    tmp[lb[0]:lb[3],lb[1]:lb[4],lb[2]:lb[5]]=1
                    gt1 = gt1*tmp
                    for gt_sl,j in zip(gt1[st:ed], range(st,ed)):
                        nz = gt_sl.nonzero()
                        lb2[j] = torch.tensor((nz[:,0].min(), nz[:,1].min(),nz[:,0].max(),nz[:,1].max()))
                # non-tight box
                else:
                    lb2 = lb[:, [1, 2, 4, 5]].repeat(s, 1)
                    lb2[:st] = 0
                    lb2[ed:] = 0

                slices = torch.linspace(st, ed, self.total_len)[
                    self.total_len // 2
                    - self.sl_len // 2 : self.total_len // 2
                    + self.sl_len // 2
                ].round().long()
                x = x[:, slices]
                lb2 = lb2[slices][None]
                # print(lb2)
                print(x.shape, lb2.shape)
            else:
                lb2=None
                print(x.shape, label.shape)
            dest = pa(save_path) / "{:05d}.pt".format(i)
            torch.save({"data": x[0][None], "label": lb2, "gt": x[1][None]}, dest)
            print("save to {}".format(dest))

    def get_whole_item(self, index, label=False, gt_mask=False):
        """
        x: N, H, W, 3
        label: 1*N * 4
        gt: 1*N*H*W"""
        index = self.data[index % len(self.data)]
        dct = torch.load(self.data_path.format(index))
        x = dct["data"].repeat(3, 1, 1, 1).permute(1, 2, 3, 0)
        y = x
        if label or gt_mask:
            y = (y,)
            if label:
                y = y + (dct["label"],)
            if gt_mask:
                y = y + (dct["gt"],)
        return y

    def __getitem__(self, index):
        """x: 3, H, W
        label: 1, (W1, H1, W2, H2)
        Note: label has (w,h) while x has (h,w)"""
        index = self.data[index // self.sl_len % len(self.data)]
        sl = index % self.sl_len
        dct = torch.load(self.data_path.format(index))
        labels = dct["label"][:, sl, [1, 0, 3, 2]]
        x = dct["data"][:, sl].repeat(3, 1, 1)

        # print(x.shape, labels)
        gt_instance = Instances((0, 0))
        gt_boxes = Boxes(labels)
        gt_instance.gt_boxes = gt_boxes
        gt_instance.gt_classes = torch.zeros(1).long()
        gt_instance.case_ind=[[index,sl]]

        # print(x.shape)
        # size = dict(height=128, width=128, depth=128)
        # x = self.normalizer(self.crop(x)).float()
        # print(x.shape, index, sl)
        return {
            "image": x,
            "instances": gt_instance,
            "height": x.size(-2),
            "width": x.size(-1),
        }


if __name__ == "__main__":
    from adet.utils.visualize_niigz import visulize_3d
    s = Slices(0)
    print("start")
    from adet.utils.visualize_niigz import draw_box, PyTMinMaxScalerVectorized
    s._prepare_data(save_path=s.data_path_nontight[:-9], tight_box=False, transpose=True)
    d, l, gt = s.get_whole_item(0, True, True)
    d = d.permute(0, 3, 1, 2)
    imgs = PyTMinMaxScalerVectorized()(d) * 256
    imgs = imgs.to(torch.uint8)

    p = draw_box(imgs, l[0])
    p = p / 255
    p = visulize_3d(p, 3, 1, save_name="ind_03_notight.png")

    gt = gt.permute(1, 0, 2, 3)
    gt = gt * 255
    gt = gt.to(torch.uint8)
    p = draw_box(gt, l[0])
    p = p / 255
    p = visulize_3d(p, 3, 1, save_name="ind_03_label_notight.png")
