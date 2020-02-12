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


def vis_match(img, match):
    n_batch = len(match)
    for i_batch in range(n_batch):
        # each match
        for m in match[i_batch]:
            nodes = []
            # each point
            for p in m:
                # draw
                nodes.append([[int(p.x+0.5), int(p.y+0.5)]])
            img[i_batch] = cv2.polylines(img[i_batch],
                                         [np.array(nodes)],  # shape [1, n, 1, 2]
                                         isClosed=True,
                                         color=(0, 191, 255),
                                         thickness=2).get()
    img = img.transpose(0, 3, 1, 2)
    return img


def draw_graphic(img, loc, score, angle, size, color, classes, show_info=False):
    img = img.copy()

    font_face = cv2.FONT_HERSHEY_COMPLEX 
    font_scale = 0.5
    thickness = 1

    # draw arrowedLine
    if angle != -1:
        rad_angle = angle / 180 * math.pi
        p2 = (int(loc[0] + 40*math.cos(rad_angle)), int(loc[1] - 40*math.sin(rad_angle)))
        img = cv2.arrowedLine(img, loc, p2, color, 2)

    # draw circle
    radius = size
    img = cv2.circle(img, loc, radius, color, 2, lineType=0)

    if show_info:
        # get text size
        if score is None:
            score = 1.0
        # # text: (x, y) | score | angle
        # text = f"({loc[0]}, {loc[1]}) | {score:.2f} | {angle:.1f}"
        # text: class_name | score
        text = f"{classes} | {score:.2f}"
        rect, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

        img = cv2.rectangle(img, (loc[0]+10, loc[1]-rect[1]+10),
                            (loc[0]+rect[0]+10, loc[1]+10), (30, 144, 255), -1)
        img = cv2.putText(img, text, (loc[0]+10, loc[1]+10),
                          font_face, font_scale, (255, 255, 255), thickness, 16)
    return img



def vis_anns(imgs, ann, classes, show_info=False):
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
            size = int(sizes[i].data[j] + 0.5)
            label = labels[i].data[j]
            color = [int(x*255) for x in _COLOR[label]]
            angle = angles[i].data[j]

            p1 = (int(locations[i].data[j, 0]), int(locations[i].data[j, 1]))
            imgs[i] = draw_graphic(imgs[i], p1, None, angle, size, color, classes[label], show_info)
    imgs = imgs.transpose(0, 3, 1, 2)

    return imgs


def vis_results(imgs, results, classes, show_info=False):
    """
    Visualize all results.

    Inputs:
        imgs: numpy images with shape [n, h, w, c].
        results -> List with shape [batch, m] (see tools/kps_tools.py)

    Outputs:
        imgs: Images will be showed with shape [n, c, h, w] with type numpy.
    """
    batch = len(results)

    for i in range(batch):
        kps = results[i]
        for idx, p in enumerate(kps):
            # angle
            angle = p.angle
            # size
            size = int(p.radius + 0.5)
            if size == -1:
                size = 20
            x, y, score = p.x, p.y, p.score
            label = p.cls
            color = [int(x*255) for x in _COLOR[label]]

            p1 = (int(x), int(y))

            imgs[i] = draw_graphic(imgs[i], p1, score, angle, size, color, classes[label], show_info)

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

