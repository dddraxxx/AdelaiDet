from matplotlib import pyplot as plt
import numpy as np
from adet.utils.dataset_3d import *
from torchvision.utils import make_grid, save_image
from pathlib import Path as pa
import torch

def save_img(path, width=5, inter_dst=10):
    img,_ = read_volume(path)
    img = torch.from_numpy(img).float()
    img = img[::inter_dst,None]
    print(img.shape, img.dtype)

    img_f = make_grid(img, nrow=width, padding=5, pad_value=-1)
    path = pa(path)
    save_image(img_f, path.with_suffix('.png'))

if __name__=='__main__':
    save_img('/home/hynx/AdelaiDet/input1.nii.gz')
    save_img('/home/hynx/AdelaiDet/output1.nii.gz')
