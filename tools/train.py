#!/usr/bin/env python
# coding=utf-8

import os
import sys
import os.path as osp
import argparse

sys.path.append('.')

import visdom
from torch.backends import cudnn

from config import cfg
from utils.logger import setup_logger
from engine.trainer import Trainer
from utils.cprint import cprint


def parse_args():
    parser = argparse.ArgumentParser(description='Get arguments from shell')

    parser.add_argument(
        '-cfg', '--config_file',
        default='',
        metavar='FILE',
        help='path to config file',
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
    if args.config_file != '':
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # gpus
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    
    # log
    work_dir = osp.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME, cfg.MODEL.VERSION)
    if not osp.exists(work_dir):
        os.makedirs(work_dir)
    logger = setup_logger('train', work_dir)

    logger.info(f"Use {num_gpus} gpus")
    logger.info(args)
    logger.info(f"Running with config:\n\n{cfg}\n")
    
    cprint('start training...')

    cudnn.benchmark = True
    
    # visdom
    vis = visdom.Visdom(port=cfg.VISDOM.PORT)
    assert vis.check_connection(), \
           'visdom server has not lauched, please run `python -m visdom.server`'
    cprint(f"Visdom is running, you can visit it at `localhost:{cfg.VISDOM.PORT}`",
           kind=1)

    trainer = Trainer(cfg, logger, vis, work_dir)
    trainer.train()


if __name__ == '__main__':
    main()

