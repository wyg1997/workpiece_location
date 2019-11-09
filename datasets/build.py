#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import DataLoader

from .transforms import build_transforms
from .image_dataset import ImageDataset
from utils.cprint import cprint


def get_dataloader(cfg):
    cprint('preparing dataset...')
    train_trans = build_transforms(cfg, is_train=True)
    test_trans = build_transforms(cfg, is_train=False)

    cprint('loading train dataset...')
    train_imgs = ImageDataset(cfg.TRAIN, train_trans, test_mode=False)
    train_dataloader = DataLoader(train_imgs,
                                  cfg.DATALOADER.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=cfg.DATALOADER.WORKERS,
                                  pin_memory=True
                                 )

    cprint('loading test dataset...')
    test_imgs = ImageDataset(cfg.TEST, test_trans, test_mode=True)
    test_dataloader = DataLoader(test_imgs,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                )

    return train_dataloader, test_dataloader, train_imgs.CLASSES
