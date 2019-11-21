#!/usr/bin/env python
# coding=utf-8

import torch

from utils.cprint import cprint
from .lr_scheduler import WarmupMultiStepLR


_ALL_OPTIMIZER = ['sgd', 'adam', 'rmsprop']
_ALL_LOSS_FUNCTION = ['mseloss']


def make_optimizer(solver_cfg, model):
    opt_name = solver_cfg.OPT.lower()

    if opt_name == 'sgd':
        opt_func = torch.optim.SGD(model.parameters(),
                                   lr=solver_cfg.BASE_LR,
                                   weight_decay=solver_cfg.WEIGHT_DECAY,
                                   momentum=solver_cfg.MOMENTUM)
    elif opt_name == 'adam':
        opt_func = torch.optim.Adam(model.parameters(),
                                    lr=solver_cfg.BASE_LR,
                                    weight_decay=solver_cfg.WEIGHT_DECAY)
    elif opt_name == 'rmsprop':
        opt_func = torch.optim.RMSprop(model.parameters(),
                                       lr=solver_cfg.BASE_LR,
                                       weight_decay=solver_cfg.WEIGHT_DECAY,
                                       momentum=solver_cfg.MOMENTUM)
    else:
        cprint(f"unknown optimizer {opt_name}, only support {_ALL_OPTIMIZER}",
               level='error')
        raise NameError(f"optimizer {opt_name} not support")

    return opt_func


def make_lr_scheduler(solver_cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        solver_cfg.STEPS,
        solver_cfg.GAMMA,
        warmup_factor=solver_cfg.WARMUP_FACTOR,
        warmup_iters=solver_cfg.WARMUP_ITERS,
        warmup_method=solver_cfg.WARMUP_METHOD
    )


def make_loss_function(loss_name):
    loss_name = loss_name.lower()

    if loss_name == 'mseloss':
        loss_func = torch.nn.MSELoss()
    else:
        cprint(f"Unknown loss function {loss_name}, only support {_ALL_LOSS_FUNCTION}",
               level='error')
        raise NameError(f"loss function {loss_name} not support")

    return loss_func
