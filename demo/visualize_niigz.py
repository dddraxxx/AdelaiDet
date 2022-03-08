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
    Note: bbox coord is re-permuted for draw_bounding_box"""
    dr_data = []
    for d, b in zip(data, bbox):
        dr_data.append(draw_bounding_boxes(d, b[[1,0,3,2]][None], colors=["red"]))
    return torch.stack(dr_data)

def visulize_3d(data, width=5, inter_dst=10, save_name=None):
    '''
    data: (S, H, W) or (N, C, H, W)'''
    img = data
    # st = torch.tensor([76, 212, 226])
    # end = st+128
    # img = img[st[0]:end[0],st[1]:end[1],st[2]:end[2]]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    if img.dim()<4:
        img = img[:,None]
    img = img[::inter_dst]
    print('Visualizing img with shape and type:', img.shape, img.dtype)

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
    save_img("/home/hynx/AdelaiDet/input1.nii.gz")
    save_img("/home/hynx/AdelaiDet/output1.nii.gz")
    # save_img(lpath.format(1), save_name="label_cropped.png")
    # save_img(dpath.format(5), save_name="data5.png")
