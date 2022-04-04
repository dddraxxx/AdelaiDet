import os
import sys
import torch
import torch.nn.functional as F
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from adet.utils.gridding import GriddingReverse
from adet.utils.chamfer_distance import ChamferDistance

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

if __name__ == '__main__':
    x = torch.rand(2, 3, 3, 3).cuda()
    x.requires_grad = True
    xx = x.unsqueeze(-1)
    xx = torch.cat([xx, 1 - xx], dim=-1)
    xx = gumbel_softmax(xx, hard=True)

    model = GriddingReverse(3).cuda()
    pc = model(xx[:, :, :, :, 0].contiguous())
    # print(pc.shape, pc.max(), pc.min())
    label = (torch.randn(pc.shape).cuda() + 1) / 2.

    loss_func = ChamferDistance().cuda()
    print(pc.shape, label.shape)
    print(pc)
    loss = loss_func(pc, label)
    print(loss.requires_grad)
    loss.backward()
    print(x.grad)


