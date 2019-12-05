#!/usr/bin/env python
# coding=utf-8

import math

import torch
import torch.nn.functional as F
import numpy as np

from utils.cprint import cprint
from utils.meters import AverageMeter


def eval_key_points(kps, anns, size=40):
    """
    Compare detection results with groundtruth.

    Inputs:
        kps -> list with shape [n, k, m, 3]
            All keypoints in heatmap with [x, y, score].
        anns -> dict('locations': <ndarray> (n, m, 2),
                     'labels': <ndarray> (n, m),
                     ...)
            Groundtruth keypoints(more informations see datasets/image_dataset.py).

    Outputs:
        avg_offset -> Float
            Average distance with groundtruth.
        precision -> Float
            Precisions with results.
        recall -> Float
            Recall with results.
    """
    locations = [x.data for x in anns['locations']]
    labels = [x.data for x in anns['labels']]
    assert len(kps) == len(locations)

    avg_offset = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    n_batch = len(kps)
    n_cls = len(kps[0])

    # for i -> batch, j -> class, i_d -> detection, i_t -> groundtruth
    for i in range(n_batch):
        for j in range(n_cls):
            dets = kps[i][j]
            tars = locations[i][labels[i] == j]

            n_dets = len(dets)
            n_tars = tars.shape[0]

            ok = 0
            visit = []
            for i_d in range(n_dets):
                for i_t in range(n_tars):
                    # continue if it have been hitted
                    if i_t in visit:
                        continue
                    if is_in_range(dets[i_d], tars[i_t], size):
                        ok += 1
                        dis = math.sqrt(
                            calc_distance_square(dets[i_d], tars[i_t]))
                        avg_offset.update(dis)
                        visit.append(i_t)
                        break
            precision.update(1, ok)
            precision.update(0, n_dets-ok)
            recall.update(1, ok)
            recall.update(0, n_tars-ok)

    return avg_offset, precision, recall


def is_in_range(p1, p2, size):
    return calc_distance_square(p1, p2) < size * size


def calc_distance_square(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def nms_points(x, y, score, size=40):
    """
    Select key points by nms.
    """

    points = []

    order = np.argsort(-score)

    for idx in order:
        p = [x[idx], y[idx], score[idx]]

        flag = True
        if points:
            for p2 in points:
                if is_in_range(p, p2, size):
                    flag = False
                    break
        if flag:
            points.append(p)

    return points


def get_kps_from_heatmap(heatmap, stride, threshold=0.5, size=40):
    """
    Calculate keypoints from heatmap.

    Input:
        heatmap -> torch with shape [n, k, h, w]
            The outputs from network.
        stride -> Int
            The stride of each pixel.
        trans_infos -> dict('do_flip': Bool, 'img_shape': [h, w], 'img_ratio:': [r_h, r_w])
            The infomations of images.
        threshold -> Float
            The threshold for bg and fg.
        size -> Int
            The size of points used in nms.

    Output:
        keypoints -> list with shape [n, k, m, 3]
            All keypoints in heatmap with [x, y, score].
    """
    keypoints = []

    if stride != 1:
        heatmap = F.interpolate(
            heatmap, scale_factor=stride, mode='bicubic', align_corners=False)
    heatmap = heatmap.numpy()

    batch, num_cls, h, w = heatmap.shape

    for i in range(batch):
        kps = []
        for j in range(num_cls):
            # attention: heatmap[n, cls, h, w] -> [y, x]
            yy, xx = np.where(heatmap[i, j] > threshold)

            # continue if res is too many
            if xx.size > h*w*0.1:
                cprint('too many result points, skipping nms...', level='debug')
                res = []
            else:
                score = heatmap[i, j, yy, xx]
                # get key points by nms
                res = nms_points(xx, yy, score, size)

            kps.append(res)
        keypoints.append(kps)

    return keypoints


if __name__ == '__main__':
    results = torch.load('temp/results.pth')

    import pickle
    with open('temp/info.pkl', 'rb') as f:
        trans_infos = pickle.load(f)

    cprint(get_kps_from_heatmap(results, 4, trans_infos), level='debug')
