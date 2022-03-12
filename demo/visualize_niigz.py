from matplotlib import pyplot as plt
import numpy as np
from adet.utils.dataset_3d import *
from torchvision.utils import make_grid, save_image, draw_bounding_boxes
from pathlib import Path as pa
import torch


def draw_box(data, bbox):
    """
    Draw a box for every slice fo data
    Data: N, C, H, W
    bbox: N, 4
    return: N, C, H, W
    Note: bbox coord is re-permuted for draw_bounding_box"""
    dr_data = []
    for d, b in zip(data, bbox):
        dr_data.append(draw_bounding_boxes(d, b[[1, 0, 3, 2]][None], colors=["red"]))
    return torch.stack(dr_data)


def draw_3d_box_on_vol(data, lb):
    """
    data: 1, S, H, W
    label: 1, 6"""
    data = (data[0] * 255).to(torch.uint8)
    lb = lb.int()
    lb2 = lb[:, [1, 2, 4, 5]].repeat(data.size(0), 1)
    lb2[: lb[0, 0]] = 0
    lb2[lb[0, 3] :] = 0
    # p = draw_box(data[lb[0, 0] : lb[0, 3], None], lb2[lb[0, 0] : lb[0, 3]])
    p = draw_box(data[:, None], lb2)
    return p / 255


def visulize_3d(data, width=5, inter_dst=10, save_name=None):
    """
    data: (S, H, W) or (N, C, H, W)"""
    img = data
    # st = torch.tensor([76, 212, 226])
    # end = st+128
    # img = img[st[0]:end[0],st[1]:end[1],st[2]:end[2]]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    if img.dim() < 4:
        img = img[:, None]
    img = img[::inter_dst]
    print("Visualizing img with shape and type:", img.shape, img.dtype)

    img_f = make_grid(img, nrow=width, padding=5, pad_value=1, normalize=True)
    if save_name:
        save_image(img_f, save_name)
    else:
        return img_f


def save_img(path, width=5, inter_dst=10, save_name=None):
    img, _ = read_volume(path)
    path = pa(path)
    visulize_3d(img, width, inter_dst, save_name or path.with_suffix(".png"))


class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """

    def __call__(self, tensor: torch.Tensor, dim=2):
        """
        tensor: N*C*H*W"""
        tensor = tensor.clone()
        s = tensor.shape
        tensor = tensor.flatten(-dim)
        scale = 1.0 / (
            tensor.max(dim=-1, keepdim=True)[0] - tensor.min(dim=-1, keepdim=True)[0]
        )
        tensor.mul_(scale).sub_(tensor.min(dim=-1, keepdim=True)[0])
        return tensor.view(*s)


if __name__ == "__main__":
    save_img("/home/hynx/AdelaiDet/input1.nii.gz")
    save_img("/home/hynx/AdelaiDet/output1.nii.gz")
    # save_img(lpath.format(1), save_name="label_cropped.png")
    # save_img(dpath.format(5), save_name="data5.png")
