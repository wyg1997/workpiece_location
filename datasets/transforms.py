#!/usr/bin/env python
# coding=utf-8

import copy

import torchvision.transforms as T
import cv2
import torch
import numpy as np
from utils.cprint import cprint


def build_transforms(cfg, is_train):
    if is_train:
        config = cfg.TRAIN
    else:
        config = cfg.TEST

    return Pipline(config, is_train)


class Pipline:
    def __init__(self, cfg, is_train):
        self.cfg = cfg
        self.is_train = is_train

    def __call__(self, results, num_cls):
        # opencv shape [h, w, c]
        img = cv2.imread(results['img_info']['filename'])
        # to rbg
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # to float
        img = img.astype(np.float32) / 255.0

        h, w = img.shape[:2]
        new_h, new_w = self.cfg.SIZE

        r_h = new_h / h
        r_w = new_w / w

        # resize img
        img = cv2.resize(img, (new_w, new_h))
        # resize labels
        ann = copy.deepcopy(results['img_info']['ann'])
        ann['locations'] = ann['locations'].astype(np.float) * [r_w, r_h];

        # TODO: do flip
        if self.cfg.DO_FLIP:
            pass

        # TODO: do lightning
        if self.cfg.DO_LIGHTNING:
            pass

        # normalize
        img = (img - self.cfg.MEAN) / self.cfg.STD

        img = img.transpose((2, 0, 1)).astype(np.float32)

        targets = self.get_gussian_targets(
                    ann,
                    new_h,
                    new_w,
                    self.cfg.STRIDE,
                    self.cfg.SIGMA,
                    num_cls
                  )

        return img, targets

    def get_gussian_target(self, center, H, W, stride, sigma):
        """
        根据一个中心点，生成高斯热力图。
        
        Inputs:
            center: 一个中心点[x, y]

        Outputs:
            target: 这个中心点的热力图。
        """
        map_h = H // stride
        map_w = W // stride

        start = stride / 2.0 - 0.5
        y_range = range(map_h)
        x_range = range(map_w)
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx*stride + start
        yy = yy*stride + start

        d2 = (xx-center[0])**2 + (yy-center[1])**2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap

    def get_gussian_targets(self, ann, H, W, stride, sigma, num_cls):
        """
        Outputs:
            heatmaps: The heatmaps of labels with shape [n, num_cls, h, w].
        """
        assert H % stride == 0 and W % stride == 0

        heatmaps = np.zeros((num_cls, H//stride, W//stride)).astype(np.float32)

        points = ann['locations']
        labels = ann['labels']
        num_points = points.shape[0]

        for i in range(num_points):
            label = labels[i]
            heatmap = self.get_gussian_target(points[i], H, W, stride, sigma)
            heatmaps[label] = np.maximum(heatmap, heatmaps[label])

        return heatmaps
