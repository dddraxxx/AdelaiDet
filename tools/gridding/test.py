import os
import sys
import torch
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from tools.gridding import GriddingReverse

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
    x = x.unsqueeze(-1)
    x = torch.cat([x, 1 - x], dim=-1)
    x = gumbel_softmax(x, hard=True)

    model = GriddingReverse(2)
    pc = model(x[:, :, :, :, 0].contiguous())
    print(pc.shape, pc.max(), pc.min())
    print(pc)
    print(x[:, :, :, :, 0] )

