#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn.functional as F

from utils.cprint import cprint


def l2_loss(output, target, mask=None):
    batch_size = output.shape[0]
    if mask is not None:
        cprint(f"output: {output[0][mask[0]].view(8, -1).permute(1, 0)[:5, :]}\n"
               f"target: {target[0][mask[0]].view(8, -1).permute(1, 0)[:5, :]}",
               level='debug')
        cprint(output[mask] - target[mask], level='debug')
        loss = (output[mask] - target[mask]) ** 2 / 2
    else:
        loss = (output - target) ** 2 / 2
    return loss.sum() / batch_size


def angle_loss_func(output, target, mask):
    batch_size = output.shape[0]

    count = mask.sum().item()
    if count == 0:
        return 0

    # debug
    torch.set_printoptions(threshold=9999998, linewidth=200)
    p_output = output.permute(0, 2, 3, 1)
    p_target = target.permute(0, 2, 3, 1)
    p_mask = mask.permute(0, 2, 3, 1)

    # # debug
    # cprint(p_output[p_mask].reshape(-1, 2)[:10], level='debug')
    # cprint(p_target[p_mask].reshape(-1, 2)[:10], level='debug')

    # loss = 1 - sin(a)*sin(x) - cos(a)*cos(x)
    loss = F.smooth_l1_loss(output[mask], target[mask], reduction='sum')
    return loss / batch_size


def size_loss_func(output, target, mask):
    batch_size = output.shape[0]

    if output[mask].sum().item() > 0:
        loss = F.smooth_l1_loss(output[mask] / target[mask],
                                torch.ones(output[mask].shape).cuda(),
                                reduction='sum')
    else:
        loss = 0

    return loss / batch_size

