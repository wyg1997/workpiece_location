#!/usr/bin/env python
# coding=utf-8

import math

import torch
import torch.nn.functional as F
from visdom import Visdom
import numpy as np
import cv2
import seaborn as sns

from utils.cprint import cprint


_COLOR = sns.color_palette('Paired', 20)


def vis_anns(imgs, ann):
    """
    Visualize annotations from dataloader.

    Inputs:
        imgs: numpy images with shape [n, c, h, w].
        ann: Annotations from dataloader with List[ann=Dict('locations'=numpy(n, 2),
                                                            'sizes'=numpy(n),
                                                            'angles'=numpy(n),
                                                            'angles'=numpy(n))]

    Outputs:
        imgs: Images will be showed with shape [n, c, h, w] with type numpy.
    """
    locations = ann['locations']
    angles = ann['angles']
    sizes = ann['sizes']
    labels = ann['labels']

    num_img = len(labels)

    for i in range(num_img):
        num_point = labels[i].data.shape[0]

        for j in range(num_point):
            size = sizes[i].data[j] * 10
            label = labels[i].data[j]
            color = [int(x*255) for x in _COLOR[label]]
            angle = angles[i].data[j] / 180.0 * math.pi

            p1 = (int(locations[i].data[j, 0]), int(locations[i].data[j, 1]))
            p2 = (int(p1[0]+size*math.cos(angle)), int(p1[1]-size*math.sin(angle)))
            imgs[i] = cv2.arrowedLine(imgs[i], p1, p2, color, 1).get()
    imgs = imgs.transpose(0, 3, 1, 2)

    return imgs


def vis_results(imgs, results):
    """
    Visualize all results.

    Inputs:
        imgs: numpy images with shape [n, h, w, c].
        results -> dict('location': keypoints, ['angle': angles])
            keypoints -> list with shape [n, k, m, 3]
                All keypoints in heatmap with [x, y, score].
            [angles -> list with shape [n, k, m, 1]]

    Outputs:
        imgs: Images will be showed with shape [n, c, h, w] with type numpy.
    """
    batch = len(results['locations'])
    num_cls = len(results['locations'][0])

    for i in range(batch):
        for j in range(num_cls):
            for idx, p in enumerate(results['locations'][i][j]):
                angle = results['angles'][i][j][idx] / 180.0 * math.pi
                x, y, score = p
                label = j
                size = 40
                color = [int(x*255) for x in _COLOR[label]]

                p1 = (int(x), int(y))
                p2 = (int(p1[0]+size*math.cos(angle)), int(p1[1]-size*math.sin(angle)))

                # 画箭头
                imgs[i] = cv2.arrowedLine(imgs[i], p1, p2, color, 1).get()
                # 写位置和角度
                pass

    return imgs.transpose(0, 3, 1, 2)


def vis_heatmaps(imgs, targets,
              alpha=0.5, threshold=0.3):
    """
    Show image and target.

    Input:
        imgs: numpy image with shape [n, h, w, c].
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

    assert imgs.shape[1] == targets.shape[-2] and \
           imgs.shape[2] == targets.shape[-1], \
           f"imgs and targets must have same shape, but get {imgs.shape} and {targets.shape}"

    targets[targets<threshold] = 0
    targets = targets.clip(min=0.0, max=1.0)
    
    num_images = imgs.shape[0]

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

    imgs = vis_heatmaps(imgs, targets, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    vis.images(imgs)

