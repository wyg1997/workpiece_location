#!/usr/bin/env python
# coding=utf-8

import torch

from utils.cprint import cprint


def l2_loss(output, target, mask=None):
    batch_size = output.shape[0]
    if mask is not None:
        cprint(f"output: {output[0][mask[0]].view(8, -1).permute(1, 0)[:5, :]}\n"
               f"target: {target[0][mask[0]].view(8, -1).permute(1, 0)[:5, :]}",
               level='debug')
        cprint(output[mask] - target[mask], level='debug')
        loss = (output[mask] - target[mask]) ** 2 / 2 / batch_size
    else:
        loss = (output - target) ** 2 / 2 / batch_size
    return loss.sum()


def angle_loss_func(output, target, mask):
    batch_size = output.shape[0]

    output = output.permute(0, 2, 3, 1)
    target = target.permute(0, 2, 3, 1)
    mask = mask.permute(0, 2, 3, 1)

    # loss = 1 - sin(a)*sin(x) - cos(a)*cos(x)
    loss = None
    for i in range(batch_size):
        if loss is None:
            loss = (1 - (output[i, mask[i]].reshape(-1, 2) * target[i, mask[i]].reshape(-1, 2)).sum(axis=1)).sum()
        else:
            loss += (1 - (output[i, mask[i]].reshape(-1, 2) * target[i, mask[i]].reshape(-1, 2)).sum(axis=1)).sum()
    return loss / batch_size

