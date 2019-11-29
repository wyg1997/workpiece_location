#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import DataLoader

from .transforms import build_transforms
from .image_dataset import ImageDataset
from utils.cprint import cprint
from datasets.collate_fn import default_collate


_ALL_KIND = ['train', 'test']

def get_dataloader(cfg, kind, CLASSES=[]):
    if len(CLASSES) == 0:
        CLASSES = None

    if kind == 'train':
        cprint('loading train dataloader...')
        train_trans = build_transforms(cfg, is_train=True)
        train_imgs = ImageDataset(cfg.TRAIN, train_trans, test_mode=False, CLASSES=CLASSES)
        train_dataloader = DataLoader(train_imgs,
                                      cfg.DATALOADER.TRAIN.BATCH_SIZE,
                                      collate_fn=default_collate,
                                      shuffle=True,
                                      num_workers=cfg.DATALOADER.TRAIN.WORKERS,
                                      pin_memory=True
                                     )
        cprint(f"finish loading, the number of images is {len(train_imgs)}")
        return train_dataloader, train_imgs.CLASSES
    elif kind == 'test':
        assert CLASSES is not None, "test dataloader `CLASSES` must not be None."
        cprint('loading test dataloader...')
        test_trans = build_transforms(cfg, is_train=False)
        test_imgs = ImageDataset(cfg.TEST, test_trans, test_mode=True, CLASSES=CLASSES)
        test_dataloader = DataLoader(test_imgs,
                                     cfg.DATALOADER.TEST.BATCH_SIZE,
                                     collate_fn=default_collate,
                                     shuffle=False,
                                     num_workers=cfg.DATALOADER.TEST.WORKERS,
                                    )
        cprint(f"finish loading, the number of images is {len(test_dataloader)}")
        return test_dataloader
    else:
        cprint(f"dataloader kind must be one of {_ALL_KIND}")
        raise NameError(f"not support dataloader kind {kind}")

