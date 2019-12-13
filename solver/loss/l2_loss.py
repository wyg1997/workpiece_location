#!/usr/bin/env python
# coding=utf-8

import torch

from utils.cprint import cprint


def l2_loss(output, target, ignore=None):
    batch_size = output.shape[0]
    if ignore is not None:
        mask = target != ignore
        loss = (output[mask] - target[mask]) ** 2 / 2 / batch_size
    else:
        loss = (output - target) ** 2 / 2 / batch_size
    return loss.sum()
