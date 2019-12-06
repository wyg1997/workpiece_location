#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn.functional as F
from visdom import Visdom
import numpy as np
import cv2

from utils.cprint import cprint


def visualize(imgs, targets=None,
              mean=[0, 0, 0], std=[1, 1, 1],
              alpha=0.5, threshold=0.3):
    """
    Show image and target.

    Input:
        imgs: numpy image with shape [n, c, h, w].
        targets: heatmaps with images with shape [n, k, h, w].
        mean: images mean.
        std: images std.
        alpha: heatmap weight.
        threshold: Ignore pixel which score is too low.

    Output:
        show_imgs: The images will be showed with shape [n, c, h, w] with type `numpy`.
    """
    assert isinstance(imgs, np.ndarray) and isinstance(targets, np.ndarray), \
           f"imgs and targets must be np.ndarray, but get {type(imgs) and {type(targets)}}"

    assert imgs.shape[-2] == targets.shape[-2] and \
           imgs.shape[-1] == targets.shape[-1], \
           f"imgs and targets must have same shape, but get {imgs.shape} and {targets.shape}"

    targets[targets<threshold] = 0
    targets = targets.clip(min=0.0, max=1.0)
    
    num_images = imgs.shape[0]

    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = (imgs*std + mean)*255
    imgs = imgs.astype(np.uint8)

    targets = targets.max(axis=1) # shape [n, h, w]
    targets = targets*255
    targets = targets.astype(np.uint8)
    targets = targets[..., np.newaxis] # shape [n, h, w, c=1]

    for i in range(num_images):
        heatmap = cv2.applyColorMap(targets[i], cv2.COLORMAP_HOT)
        imgs[i] = cv2.addWeighted(heatmap, alpha, imgs[i], 1-alpha, 0)
    imgs = imgs.transpose(0, 3, 1, 2)

    return imgs


if __name__ == '__main__':
    vis = Visdom(port=8887)
    imgs = torch.load('temp/imgs.pth')
    targets = torch.load('temp/targets.pth')

    imgs = visualize(imgs, targets, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    vis.images(imgs)

