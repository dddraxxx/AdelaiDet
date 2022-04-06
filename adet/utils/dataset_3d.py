from functools import reduce
from random import randrange
from typing import Tuple
import numpy as np

import torch.nn.functional as F

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


dpath = "/mnt/sdc/kits21/data/case_00{:03d}/imaging.nii.gz"
lpath = "/home/hynx/kits21/kits21/data/case_00{:03d}/aggregated_MAJ_seg.nii.gz"


def fd(a, b):
    return torch.div(a, b, rounding_mode="floor")


torch.Tensor.__floordiv__ = fd


class Boxes3D(Boxes):
    def __init__(self, tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2,x3, y3).
        """
        device = (
            tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        )
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 6)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 6, tensor.size()

        self.tensor = tensor

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes3D(self.tensor.to(device=device))

    def clip(self, box_size: Tuple[int, int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        s, h, w = box_size
        x1 = self.tensor[:, 2].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        s1 = self.tensor[:, 0].clamp(min=0, max=s)
        x2 = self.tensor[:, 5].clamp(min=0, max=w)
        y2 = self.tensor[:, 4].clamp(min=0, max=h)
        s2 = self.tensor[:, 3].clamp(min=0, max=s)
        self.tensor = torch.stack((s1, y1, x1, s2, y2, x2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 3] - box[:, 0]
        heights = box[:, 4] - box[:, 1]
        depth = box[:, 5] - box[:, 2]
        keep = (widths > threshold) & (heights > threshold) & (depth > threshold)
        return keep

    def scale(self, scale_x: float, scale_y: float, scale_z: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::3] *= scale_x
        self.tensor[:, 1::3] *= scale_y
        self.tensor[:, 2::3] *= scale_z

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        keep = self.nonempty()
        area = (
            (box[:, 3] - box[:, 0]) * (box[:, 4] - box[:, 1]) * (box[:, 5] - box[:, 2])
        )
        area[~keep] = 0
        return area

    def iou(self, id1, id2) -> torch.Tensor:
        inter_box = torch.cat(
            [
                torch.max(self.tensor[id1, :3], self.tensor[id2, :3]),
                torch.min(self.tensor[id1, 3:], self.tensor[id2, 3:]),
            ],
            dim=-1,
        )
        # print('from box3d iou')
        # print(inter_box.shape)
        i = Boxes3D(inter_box).area()
        u = self[id1].area() + self[id2].area() - i
        return i / u

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes3D(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert (
            b.dim() == 2
        ), "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes3D(b)


def read_header(path):
    image = sitk.ReadImage(path)
    header = {
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
    }
    return header


def read_volume(path):
    '''
    return 3d numpy array and its header'''
    path = str(path)
    image = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(image)
    header = {
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
    }
    return data, header


def save_volume(path, data, header):
    """
    CAREFUL you need to restore_original_slice_orientation before saving!
    :param img:
    :param header:
    :return:
    """
    path = str(path)
    img_itk = sitk.GetImageFromArray(data)

    img_itk.SetSpacing(header["spacing"])
    img_itk.SetOrigin(header["origin"])
    if not isinstance(header["direction"], tuple):
        img_itk.SetDirection(header["direction"].flatten())
    else:
        img_itk.SetDirection(header["direction"])

    sitk.WriteImage(img_itk, path)


def draw_edge(data, mx, mi):
    m = np.stack([mx, mi], axis=1)

    # print(m)

    for i in range(3):
        edge = np.s_[mi[i] : mx[i]]
        coord = [
            None,
        ] * 3
        coord[i] = [edge]
        coord[(i + 1) % 3] = m[(i + 1) % 3]
        coord[(i + 2) % 3] = m[(i + 2) % 3]
        for c in tuple(product(*coord)):
            data[c] = 2


def get_label(data, label_no=1, thres=20):
    """
    data: S, H, W
    labels: N*6"""
    data = (data == label_no).astype(int)
    ldata, n = label(data, np.ones((3, 3, 3)))
    ls = find_objects(ldata)
    rmd = []
    # filter small segments
    for inst in ls:
        for sl in inst:
            if sl.stop - sl.start < thres:
                rmd.append(inst)
                break
    for r in rmd:
        ls.remove(r)
    mis = torch.tensor([[i.start for i in sl] for sl in ls])
    mxs = torch.tensor([[i.stop for i in sl] for sl in ls])
    bbox = torch.cat([mis, mxs], dim=1)  # [:,[0,3,1,4,2,5]]
    return bbox


def get_dataset(length):
    return Volumes(length)


def pad_to(coord, size):
    ctr = (coord[:3] + coord[3:]) // 2
    size = torch.tensor(size)
    return torch.cat([ctr - size // 2, ctr + size // 2])


def normalizer_3d(x):
    return (x - x.mean(dim=[1, 2, 3], keepdim=True)) / x.std(
        dim=[1, 2, 3], keepdim=True
    )


class Copier(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)


class Volumes(Dataset):
    def __init__(self, length):
        super().__init__()
        self.length = length
        # 10 samples total
        self.data = list(range(240))  # [0, 1, 2, 3, 4] 5, 6, 7, 8, 9]
        if 236 in self.data:
            self.data.remove(236)
        if 296 in self.data:
            self.data.remove(296)
        # case236, case296 has no label
        self.prep_data = list(range(0, 300))
        self.crop_size = (128,) * 3
        # self.crop = T.RandSpatialCrop(
        #     (128,128,128), random_center=False, random_size=False
        # )
        self.normalizer = normalizer_3d
        # self.header = read_header(dpath.format(self.data[0]))
        self.header = {
            "spacing": (0.5, 0.919921875, 0.919921875),
            "origin": (0.0, 0.0, 0.0),
            "direction": (-0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0),
        }
        self.ppr_datapath = "/mnt/sdc/kits21/data_3d_ppr/{:05d}.pt"
        self.orig_datapath = "/mnt/sdc/kits21/data_3d/{:05d}.pt"
        self.datapath = self.ppr_datapath

    def _prepare_data(self, save_path="/mnt/sdc/kits21/data_3d_ppr"):
        """
        data: 1, S, H, W
        gt: 1,S,H,W
        label: 1, 6"""
        for i in self.prep_data:
            x, label = self.read_data(i, read_gt=True, preprocess=False)
            gt = x[1:]
            if len(label) != 0:
                # data, label = self.preprocess(x[:1], label)
                data, gt, label = self.preprocess(x, label, include_gt=True)
                assert data.dim() == 4, label.dim() == 2
            else:
                data = x[0][None]
            dest = pa(save_path) / "{:05d}.pt".format(i)
            torch.save({"data": data, "label": label, "gt": gt}, dest)
            print("{}: save to {}".format(i, dest))

    def preprocess(self, data, label, include_gt=False):
        """
        Get the most left label and resize_crop it to crop_size
        data: 1, S, H, W
        gt (binary mask): 1, S, H, W
        label: N, 6"""
        # pick the most left one
        l = label[[0]].long()

        from visualize_niigz import PyTMinMaxScalerVectorized

        if include_gt:
            gt = data[1:]
            c = l[0]
            zs = torch.zeros_like(gt)
            zs[..., c[0] : c[3], c[1] : c[4], c[2] : c[5]] = 1
            gt = gt * zs
            data = data[:1]
        if not include_gt:
            gt = data.new_zeros(data.shape)
            gt[:, l[0, 0] : l[0, 3], l[0, 1] : l[0, 4], l[0, 2] : l[0, 5]] = 1

        # normalize it
        # print(data.shape, data.min(), data.view(-1).mode())
        data = self.normalizer(data)
        data.clamp_(-3, 3)
        data = PyTMinMaxScalerVectorized()(data, 3)
        bkgr_value = data.min()

        # print(gt.shape, l)

        # # Resize image so that label area is small
        length = l[:, [3, 4, 5]] - l[:, [0, 1, 2]]
        # print(length)
        mil, mal = length.min(), length.max()
        if mal / 64 < mil / 16:
            factor = mal / 64
        else:
            factor = min(mil / 16, mal / 96)
        rs_data = F.interpolate(
            data[None],
            scale_factor=1 / factor.item(),
            mode="trilinear",
            align_corners=True,
        )[0]
        rs_gt = F.interpolate(
            gt[None],
            scale_factor=1 / factor.item(),
            mode="trilinear",
            align_corners=True,
        )[0]
        rs_gt = rs_gt > 0.5

        # # Crop it to crop_size containing the label area
        # (start,end, start,end, ...)
        rs_l = torch.from_numpy(T.BoundingRect()(rs_gt)[0])
        rs_ll = (rs_l - rs_l.roll(1, 0))[1::2] / 4
        sel_rge = torch.stack([rs_ll.ceil(), (rs_ll * 3).floor()]).t().long()
        # print(rs_l, rs_ll, sel_rge)
        rand = lambda x: randrange(x[0], x[1])
        sel_shift = torch.tensor([rand(rge.tolist()) for rge in sel_rge])
        sel_p = sel_shift + rs_l[0::2]
        sel_p = sel_p.int()
        # print(sel_shift, sel_p)
        # -//2
        cr_data = T.SpatialCrop(sel_p, self.crop_size)(rs_data)
        cr_gt = T.SpatialCrop(sel_p, self.crop_size)(rs_gt)

        # print(cr_data.shape)
        if any([i < j for i, j in zip(cr_data.shape, self.crop_size)]):
            # +//2, +//2+1
            cr_data = T.SpatialPad(
                self.crop_size,
                method="symmetric",
                mode="constant",
                value=data.view(-1).mode()[0],
            )(cr_data)
            cr_gt = T.SpatialPad(
                self.crop_size,
                method="symmetric",
                mode="constant",
                value=bkgr_value,
            )(cr_gt)
        # print(cr_data.shape, cr_gt.shape)
        cr_l = T.BoundingRect()(cr_gt)[0][[0, 2, 4, 1, 3, 5]]
        # print(cr_l)
        if include_gt:
            return cr_data, cr_gt, cr_l
        return cr_data, cr_l

    def read_data(self, ind, read_gt=False, transpose=False, preprocess=True):
        """
        Read from nii.gz files. Old version -- normalize + crop...
        data: 1*S*H*W
        label: 1*6"""
        data = read_volume(dpath.format(ind))[0][None]

        # print(ind)
        data = torch.as_tensor(data)
        # data = self.normalizer(data)
        # 1 is kidney (I think)
        labels = get_label(read_volume(lpath.format(ind))[0], 1)

        if read_gt:
            gt = torch.as_tensor(read_volume(lpath.format(ind))[0][None])
            data = torch.cat([data, gt], dim=0)

        if transpose:
            # Transpose shape to make each slice same size
            data = torch.einsum("...shw->...wsh", data)
            labels = labels[..., [2, 0, 1, 5, 3, 4]] if len(labels) else labels

        # Try to make data smaller to
        if self.crop_size and preprocess:
            # print("crop data")
            # print(labels)
            # padding data
            if any(i < j for i, j in zip(data.shape[-3:], self.crop_size)):
                s, h, w = data.shape[-3:]
                cs = self.crop_size
                f = lambda x: (x // 2, x // 2 + x % 2) if x > 0 else (0, 0)
                pads = [f(i) for i in (cs[2] - w, cs[1] - h, cs[0] - s)]
                pads = reduce(lambda x, y: x + y, pads)
                # print("padding data:{} for shape {}".format(pads, data.shape[-3:]))
                data = F.pad(data, pads)
                labels = labels + torch.tensor(pads)[::2].flip(-1).repeat(2)

            label = labels[0]
            # print(label)
            coord = pad_to(label, self.crop_size)
            # print(coord, data.shape)
            coord = coord.view(2, 3).T
            for i in range(3):
                if coord[i][0] < 0:
                    coord[i] = coord[i] - coord[i][0]
                if coord[i][1] > data.shape[-3:][i]:
                    coord[i] = coord[i] - (coord[i][1] - data.shape[-3:][i])
            coord = coord.T.view(-1)

            # print("data index:{}".format(ind))
            # print("crop coodinate:{}".format(coord))

            def crop(data, coord):
                data = data[
                    ...,
                    coord[0] : coord[3],
                    coord[1] : coord[4],
                    coord[2] : coord[5],
                ]
                return data

            data = crop(data, coord)
            assert data.shape[-3:] == (128, 128, 128)

            label[:3] = torch.maximum(label[:3] - coord[:3], torch.zeros(3))
            label[3:] = torch.minimum(label[3:] - coord[:3], coord[3:] - coord[:3])

            labels = label[None]
        # print(label)

        return data.float(), labels.float()

    def get_data(self, index, read_gt=False):
        """
        data: 1, S, H, W
        gt: 1, S, H, W
        label: 1, 6"""
        dct = torch.load(self.datapath.format(index))
        x, labels = dct["data"], dct["label"]
        if not read_gt:
            return x.float(), torch.as_tensor(labels)[None].float()
        else:
            return x.float(), torch.as_tensor(labels)[None].float(), dct["gt"]

    def __getitem__(self, index):
        """
        Do normalize and crop after getting item.
        data: 1*s*h*w
        label: 1*6 (s1,h1,w1,s2,h2,w2)"""
        index = self.data[index % len(self.data)]
        # Old way
        # x, labels = self.read_data(index)

        # New way
        # x, labels = self.get_data(index)
        x, labels, gt = self.get_data(index, read_gt=True)
        # print('using im_id {}'.format(index))
        # print(x.shape, labels)
        gt_instance = Instances((0, 0))
        gt_boxes = Boxes3D(labels)
        gt_instance.gt_boxes = gt_boxes
        gt_instance.gt_classes = torch.zeros(1).long()
        gt_instance.gt_masks = gt

        # print(x.shape)
        # size = dict(height=128, width=128, depth=128)
        x = self.normalizer(x).float()
        return {
            "image": x,
            "instances": gt_instance,
            "height": self.crop_size[0],
            "index": index,
        }

    def __len__(self):
        return self.length


def demo_plot(data, label):
    label = torch.min(label, torch.ones_like(label) * torch.tensor(data.shape))
    mis = label[:, :3].int().tolist()
    mxs = label[:, 3:].int().tolist()

    for mx, mi in zip(mxs, mis):
        draw_edge(data, mx, mi)

    # save_volume('test.nii.gz',data, header)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    x, y, z = data.nonzero()
    ax = plt.axes(projection="3d")
    ax.scatter3D(x, y, z, c=data[x, y, z].ravel())
    plt.savefig("demo.jpg")


def only_get_label(ind):
    labels = get_label(read_volume(lpath.format(ind))[0], 1)
    return labels


if __name__ == "__main__":
    # all_l = []
    # for i in range(5,6):
    #     labels = only_get_label(i)
    #     on_diff_side =  labels[0,0]<256
    #     if not on_diff_side:
    #         print(labels)
    #     all_l.append(labels)
    # print(len(all_l))
    d = Volumes(10)
    print("start")
    d._prepare_data()
    # d._prepare_data(save_path=d.orig_datapath[:-9])
    # d.datapath = d.orig_datapath

    # mi = 1
    # for i in d.data:
        # _, l = d.read_data(i, preprocess=False, read_gt=True)
        # gt = _[1]
        # _, l, gt = d.get_data(i, read_gt=True)
        # print(i, l)
        # assert len(l.view(-1)) > 0
        # la = (l[0, 3:] - l[0, :3]).prod()
        # ga = (gt == 1).sum()
        # print(la, ga, ga / la)

    # print(_.view(-1).mode())
    # print(l[:, [3, 4, 5]] - l[:, [0, 1, 2]], _.shape)

    # visualize
    from adet.utils.visualize_niigz import (
        draw_box,
        visulize_3d,
        PyTMinMaxScalerVectorized,
    )

    # v1
    # data, lb = d.read_data(0, read_gt=True, preprocess=False)
    # lb = lb[[0]]
    # print("normalize")
    # data = d.normalizer(data)
    # data.clamp_(-3, 3)
    # data = (PyTMinMaxScalerVectorized()(data[0], 3) * 255).to(torch.uint8)
    # gt = data[1]
    # v2
    data, lb, gt = d.get_data(0, read_gt=True)
    data = (data[0] * 255).to(torch.uint8)
    gt = gt[0]

    print("get_data")
    lb = lb.int()
    lb2 = lb[:, [1, 2, 4, 5]].repeat(data.size(0), 1)
    lb2[: lb[0, 0]] = 0
    lb2[lb[0, 3] :] = 0
    # show label in data
    p = draw_box(data[lb[0, 0] : lb[0, 3], None], lb2[lb[0, 0] : lb[0, 3]])
    p = draw_box(data[:, None], lb2)
    # demo_plot(data[1].numpy(), lb)
    visulize_3d(p / 255, 3, 5, save_name="01data_3d.png")
    # show color bar
    # p = data[lb[0, 0] : lb[0, 3]].float()
    # p = visulize_3d(p , 5, 10)
    # import matplotlib.pyplot as plt
    # plt.imshow(p.permute(1,2,0))
    # plt.colorbar()
    # plt.savefig('01data_3d.png')

    gt = gt * 255
    gt = gt.to(torch.uint8)
    p = draw_box(gt[lb[0, 0] : lb[0, 3], None], lb2[lb[0, 0] : lb[0, 3]])
    p = draw_box(gt[:, None], lb2)
    p = p / 255
    p = visulize_3d(p, 3, 5, save_name="01gt_3d.png")

    # check shape
    # for i in range(10):
    #     print(d[i]["image"].shape)
