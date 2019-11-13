#!/usr/bin/env python
# coding=utf-8

import torch
from bisect import bisect_right

from utils.cprint import cprint


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    _ALL_WARMUP_METHOD = ['constant', 'linear']
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0/3,
        warmup_iters=500,
        warmup_method='linear',
        last_epoch=-1
    ):
        if warmup_method not in self._ALL_WARMUP_METHOD:
            cprint(f"unknown warmup_method {warmup_method}, "
                   f"only support {self._ALL_WARMUP_METHOD}")
            raise NameError(f"warmup_method {warmup_method} not support")

        if milestones == sorted(milestones):
            self.milestones = milestones
        else:
            cprint(f"milestones should be a sorted list", level='warn')
            self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1-alpha) + alpha

        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]



