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

dpath = "/home/hynx/kits21/data/case_000{:02d}/imaging.nii.gz"
lpath = "/home/hynx/kits21/kits21/data/case_000{:02d}/aggregated_MAJ_seg.nii.gz"


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
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 6, tensor.size()

        self.tensor = tensor

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes3D(self.tensor.to(device=device))


def read_volume(path):
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


def get_label(data, label_no=1):
    data = (data == label_no).astype(int)
    ldata, n = label(data, np.ones((3, 3, 3)))
    ls = find_objects(ldata)
    rmd = []
    # filter small segments
    for inst in ls:
        for sl in inst:
            if sl.stop - sl.start < 20:
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


class Volumes(Dataset):
    def __init__(self, length):
        super().__init__()
        self.length = length
        # 10 samples total
        self.data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.crop_size = (128,) * 3
        # self.crop = T.RandSpatialCrop(
        #     (128,128,128), random_center=False, random_size=False
        # )
        self.normalizer = lambda x: (x - x.mean(dim=[1, 2, 3], keepdim=True)) / x.std(
            dim=[1, 2, 3], keepdim=True
        )

    def read_data(self, ind):
        data = read_volume(dpath.format(ind))[0][None]
        # print(ind)
        data = torch.as_tensor(data)
        # 1 is kidney (I think)
        labels = get_label(read_volume(lpath.format(ind))[0], 1)

        if self.crop_size:
            # print(labels)
            # padding data
            if any(i < j for i, j in zip(data.shape[-3:], self.crop_size)):
                s, h, w = data.shape[-3:]
                cs = self.crop_size
                f = lambda x: (x // 2, x // 2 + x % 2) if x>0 else (0,0)
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

    def __getitem__(self, index):
        index = self.data[index % len(self.data)]
        x, labels = self.read_data(index)
        # print(x.shape, labels)
        gt_instance = Instances((0, 0))
        gt_boxes = Boxes3D(labels)
        gt_instance.gt_boxes = gt_boxes

        x = self.normalizer(x)

        # print(x.shape)
        # size = dict(height=128, width=128, depth=128)
        # x = self.normalizer(self.crop(x)).float()
        return {"image": x, "instances": gt_instance}

    def __len__(self):
        return self.length


def demo_plot():
    path = ["/home/hynx/kits21/kits21/data/case_00020/aggregated_MAJ_seg.nii.gz"]

    data, header = read_volume(path)

    data = (data == 1).astype(int)

    ldata, n = label(data, np.ones((3, 3, 3)))
    ls = find_objects(ldata)
    mis = [[i.start for i in sl] for sl in ls]
    mxs = [[i.stop for i in sl] for sl in ls]

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
    d = Volumes(100)
    for i in range(4, 10):
        print(d[i]["image"].shape)
        print(d[i]["image"].min(),d[i]["image"].max())
