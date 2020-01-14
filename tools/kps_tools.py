#!/usr/bin/env python
# coding=utf-8

import math

import torch
import torch.nn.functional as F
import numpy as np

from utils.cprint import cprint
from utils.meters import AverageMeter
from utils.points import Point


def resize_heatmaps(heatmaps, stride):
    """
    Resize heatmaps from tensor in cuda.
    
    Inputs:
        heatmaps: The network output heatmaps with shape [n, c, h, w].
        stride: The stride between origin images and heatmaps.

    Outputs:
        heatmaps: Result heatmaps with numpy.
    """
    # resize
    if stride != 1:
        heatmaps = F.interpolate(
            heatmaps, scale_factor=stride, mode='bicubic', align_corners=False)

    # to cpu
    if heatmaps.is_cuda:
        heatmaps = heatmaps.cpu()

    # to numpy
    heatmaps = heatmaps.detach().numpy()
    return heatmaps


def eval_key_points(res, anns, size=40):
    """
    Compare detection results with groundtruth.

    Inputs:
        res -> List with shape [batch, m].
            `m` means number of keypoints defined in `utils/points.py`
        anns -> dict('locations': <ndarray> (n, m, 2),
                     'labels': <ndarray> (n, m),
                     ...)
            Groundtruth keypoints(more informations see datasets/image_dataset.py).

    Outputs:
        dict('dis': offset, 'precision': precision, 'recall': recall,
             ['angle_error': angle_error], ['size_error': size_error])
            offset -> Float
                Distance with groundtruth.
            precision -> Float
                Precisions with results.
            recall -> Float
                Recall with results.
            [angle_error -> Float
                Angle offset with groundtruth]
            [size_error -> Float
                Size offset with groundtruth]
    """
    # groundtruth
    locations = [x.data for x in anns['locations']]
    labels = [x.data for x in anns['labels']]
    angles = [x.data for x in anns['angles']]
    sizes = [x.data for x in anns['sizes']]

    # eval results
    precision = AverageMeter()
    recall = AverageMeter()
    offset = AverageMeter()
    angle_error = AverageMeter()
    size_error = AverageMeter()

    n_batch = len(res)
    # for i -> batch, j -> kps, k -> groundtruth
    for i in range(n_batch):
        kps = res[i]
        tar_locations = locations[i]
        tar_angles = angles[i]
        tar_sizes = sizes[i]
        tar_labels = labels[i]

        vis = []
        ok = 0

        n_kps = len(kps)
        n_gts = len(tar_locations)
        for j in range(n_kps):
            p = kps[j]
            for k in range(n_gts):
                # different class, continue
                if k in vis or tar_labels[k] != p.cls:
                    continue
                if is_in_range((p.x, p.y), tar_locations[k], size):
                    ok += 1
                    # dis
                    dis = math.sqrt(calc_distance_square((p.x, p.y), tar_locations[k]))
                    offset.update(dis)
                    # angle
                    if p.angle != -1:
                        angle_off = abs(p.angle - tar_angles[k])
                        angle_error.update(min(angle_off, 360-angle_off))
                    # size
                    if p.radius != -1:
                        size_error.update(abs(p.radius - tar_sizes[k]))
                    vis.append(k)
                    break
        precision.update(1, ok)
        precision.update(0, n_kps-ok)
        recall.update(1, ok)
        recall.update(0, n_gts-ok)

    return dict(dis=offset, precision=precision, recall=recall,
                angle_error=angle_error, size_error=size_error)


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
    score = score.clip(0.0, 1.0)

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


def calc_angle(v_sin, v_cos):
    a = math.atan(v_sin/(v_cos+1e-10)) * 180 / math.pi
    if v_cos < 0:
        a += 180
    # [0, 360)
    a = a % 360
    return a


def error_exclude(list_sin, list_cos):
    a_value = []
    for i in range(4):
        v_sin = list_sin[i]
        v_cos = list_cos[i]

        angle_value = calc_angle(v_sin, v_cos) + 45 + i*90
        angle_value = angle_value % 360

        list_sin[i] = math.sin(angle_value/180*math.pi)
        list_cos[i] = math.cos(angle_value/180*math.pi)
        a_value.append(angle_value)

    # error excluding algorithm
    for i in range(len(a_value)):
        a_rest = []
        for j in range(len(a_value)):
            if j != i:
                a_rest.append(a_value[j])
        x1 = abs(a_value[i] - a_rest[0])
        x2 = abs(a_value[i] - a_rest[1])
        x3 = abs(a_value[i] - a_rest[2])
        delta_angle = max(x1, x2, x3) - min(x1, x2, x3)
        if x1 > delta_angle and x2 > delta_angle and x3 > delta_angle:
            if  delta_angle < 60:
                del list_sin[i]
                del list_cos[i]
                break

    sin_value = sum(list_sin) / len(list_sin)
    cos_value = sum(list_cos) / len(list_cos)

    angle_value = calc_angle(sin_value, cos_value)
    return angle_value


def get_kps_from_heatmap(results, threshold=0.5, size=40):
    """
    Calculate keypoints from network output.

    Input:
        heatmap -> Dict('location': np.array[n, k, h, w],
                        ['angles': np.array[n, 8, h, w]],
                        ['sizes': np.array[n, 1, h, w]])
            The outputs from network.
        threshold -> Float
            The threshold for bg and fg.
        size -> Int
            The size of points used in nms.

    Output:
        keypoints -> List with shape [batch, m].
            `m` means number of keypoints defined in `utils/points.py`
    """
    keypoints = []

    heatmap = results['locations']
    batch, num_cls, h, w = heatmap.shape
    # angle map
    if 'angles' in results:
        angle_map = results['angles']
    # size map
    if 'sizes' in results:
        size_map = results['sizes']

    # points
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

            # save results
            for p in res:
                x, y, score = p
                cls = j
                # angle
                if 'angles' in results:
                    list_sin = []
                    list_cos = []
                    for b in range(4):
                        v_sin = angle_map[i, b*2, y, x]
                        v_cos = angle_map[i, b*2+1, y, x]
                        list_sin.append(v_sin)
                        list_cos.append(v_cos)
                    a = error_exclude(list_sin, list_cos)
                else:
                    a = -1
                # size
                if 'sizes' in results:
                    r = size_map[i, 0, y, x]
                else:
                    r = -1
                point = Point(x=x, y=y, score=score, radius=r, angle=a, cls=cls, from_net=True)
                kps.append(point)
        keypoints.append(kps)

    return keypoints

