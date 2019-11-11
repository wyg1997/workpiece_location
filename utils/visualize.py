#!/usr/bin/env python
# coding=utf-8

import torch
from visdom import Visdom
import numpy as np
import cv2


_color_map = [
    'Viridis',
    'Plasma',
    'Inferno',
    'Magma',
]

def visualize(imgs, targets=None, mean=[0, 0, 0], std=[1, 1, 1], alpha=0.5):
    """
    Show image and target.

    Input:
        imgs: torch image with shape [n, c, w, h].
        targets: heatmaps with images with shape [n, k, w, h],
            k is the number of classes include background.
        mean: images mean.
        std: images std.
        alpha: heatmap weight.

    Output:
        show_imgs: The images will be showed with shape [n, c, w, h] with type `numpy`.
    """
    assert imgs.shape[0] == targets.shape[0], \
            'the number of images and targets must be same'
    
    # tensor to numpy
    imgs = imgs.numpy()
    targets = targets.numpy()

    num_images = imgs.shape[0]

    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = (imgs*std + mean)*255
    imgs = imgs.astype(np.uint8)

    targets = targets.max(axis=1) # shape [n, w, h]
    targets = targets*255
    targets = targets.astype(np.uint8)
    targets = targets[..., np.newaxis] # shape [n, w, h, c=1]

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

