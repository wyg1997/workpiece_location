#!/usr/bin/env python
# coding=utf-8

import os
import sys
import os.path as osp
import argparse

sys.path.append('.')

import torch
import visdom
from torch.backends import cudnn

from config import cfg
from engine.tester import Tester
from utils.cprint import cprint
from utils.logger import setup_logger
from models.build import build_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-cfg', '--config_file',
        required=True,
        metavar='FILE',
        help='path to config file',
        type=str
    )
    parser.add_argument(
        '-cpt', '--checkpoint',
        required=True,
        metavar='FILE',
        help='path to checkpoint which will be loaded',
        type=str
    )
    parser.add_argument(
        'opts',
        help='Modify the configs from command line',
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # get config
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.PRETRAIN = False
    cfg.freeze()

    # gpus
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    cudnn.benchmark = True

    # get and check work_dir
    work_dir = osp.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME, cfg.MODEL.VERSION)
    if not osp.exists(work_dir):
        cprint(f"{work_dir} is not exists!!!", level='warn')
        op = input(f"work_dir not exists, continue testing? [y/n]")
        if op[0] == 'y':
            os.makedirs(work_dir)
        else:
            exit(0)
    # logger
    logger = setup_logger('test', work_dir)
    # visdom
    vis = visdom.Visdom(port=cfg.VISDOM.PORT, env='test')
    assert vis.check_connection(), \
           'visdom server has not lauched, please run `python -m visdom.server`'
    cprint(f"Visdom is running, you can visit it at `localhost:{cfg.VISDOM.PORT}`",
           kind=1)

    print()
    cprint(f"model description: {cfg.MODEL.DESCRIPTION}", kind=1)
    print()

    cprint('start testing...')
    cprint(f"load checkpoint at `{args.checkpoint}`", kind=1)

    # prepare testing parameters
    cpt = torch.load(args.checkpoint)
    classes = cpt['classes']
    num_cls = len(classes)
    model = build_model(cfg.MODEL.BACKBONE, num_cls, cfg.MODEL.PRETRAIN)
    model.load_state_dict(cpt['checkpoint'])
    model = model.cuda()

    tester = Tester(cfg, logger, vis, work_dir, model, classes)
    tester.test(threshold=0.5, show=True)


if __name__ == '__main__':
    main()
