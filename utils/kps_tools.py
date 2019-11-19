#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn.functional as F
import numpy as np

from cprint import cprint


def calc_distance_square(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def nms_points(x, y, score, size=40):
    """
    Select key points by nms.
    """
    def _is_in_range(p1, p2, size):
        return calc_distance_square(p1, p2) < size * size

    points = []

    order = np.argsort(-score)

    for idx in order:
        p = [x[idx], y[idx], score[idx]]

        flag = True
        if points:
            for p2 in points:
                if _is_in_range(p, p2, size):
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
        keypoints -> list with shape [n, k-1, m, 3]
            All keypoints in heatmap with [x, y, score].
    """
    keypoints = []

    if stride != 1:
        heatmap = F.interpolate(
            heatmap, scale_factor=stride, mode='bicubic', align_corners=False)
    heatmap = heatmap.numpy()

    batch = heatmap.shape[0]
    num_cls = heatmap.shape[1] - 1

    for i in range(batch):
        kps = []
        for j in range(1, num_cls + 1):
            xx, yy = np.where(heatmap[i, j] > threshold)
            score = heatmap[i, j, xx, yy]

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
