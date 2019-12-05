#!/usr/bin/env python
# coding=utf-8


def l2_loss(output, target):
    batch_size = output.shape[0]
    loss = (output - target) ** 2 / 2 / batch_size
    return loss.sum()
