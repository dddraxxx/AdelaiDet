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
        # self.data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.data=[2]
        self.sl_len = 6
        self.total_len = 10
        self.crop_size = None
        # self.crop = T.RandSpatialCrop(
        #     (128,128,128), random_center=False, random_size=False
        # )
        # self.normalizer = lambda x: (x - x.mean(dim=[1, 2, 3], keepdim=True)) / x.std(
        #     dim=[1, 2, 3], keepdim=True
        # )
        self.data_path = "/mnt/sdc/kits21/data_2d/{:05d}.pt"

    def _prepare_data(self, num_slices=None, save_path="/mnt/sdc/kits21/data_2d"):
        """
        data, gt: 1,N,H,W
        label: 1, N, 4"""
        nunm_sl = num_slices or self.sl_len
        for i in self.data:
            x, label = self.read_data(i, read_gt=True)
            s = x.size(1)

            # print(label.unique(), label.shape)
            # read_data has read all the label
            lb = label[:1]
            lb = lb.int()
            lb2 = lb[:, [1, 2, 4, 5]].repeat(s, 1)
            st = lb[0, 0]
            ed = lb[0, 3]
            lb2[:st] = 0
            lb2[ed:] = 0

            slices = torch.linspace(st, ed, self.total_len)[
                self.total_len // 2
                - self.sl_len // 2 : self.total_len // 2
                + self.sl_len // 2
            ].round().long()
            x = x[:, slices]
            lb2 = lb2[slices]
            print(x.shape, lb2.shape)
            dest = pa(save_path) / "{:05d}.pt".format(i)
            torch.save({"data": x[0][None], "label": lb2[None], "gt": x[1][None]}, dest)
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

        # print(x.shape)
        # size = dict(height=128, width=128, depth=128)
        # x = self.normalizer(self.crop(x)).float()
        return {
            "image": x,
            "instances": gt_instance,
            "height": x.size(-2),
            "width": x.size(-1),
        }


class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """

    def __call__(self, tensor: torch.Tensor):
        """
        tensor: N*C*H*W"""
        s = tensor.shape
        tensor = tensor.flatten(-2)
        scale = 1.0 / (
            tensor.max(dim=-1, keepdim=True)[0] - tensor.min(dim=-1, keepdim=True)[0]
        )
        tensor.mul_(scale).sub_(tensor.min(dim=-1, keepdim=True)[0])
        return tensor.view(*s)


if __name__ == "__main__":
    from demo.visualize_niigz import visulize_3d
    s = Slices(0)
    print("start")
    from demo.visualize_niigz import draw_box
    # s._prepare_data()
    d, l, gt = s.get_whole_item(2, True, True)
    d = d.permute(0, 3, 1, 2)
    imgs = PyTMinMaxScalerVectorized()(d) * 256
    imgs = imgs.to(torch.uint8)

    p = draw_box(imgs, l[0])
    p = p / 255
    p = visulize_3d(p, 3, 1, save_name="ind_01.png")

    gt = gt.permute(1, 0, 2, 3)
    gt = gt * 255
    gt = gt.to(torch.uint8)
    p = draw_box(gt, l[0])
    p = p / 255
    p = visulize_3d(p, 3, 1, save_name="ind_01_label.png")
