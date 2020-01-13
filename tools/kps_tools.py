#!/usr/bin/env python
# coding=utf-8

import math

import torch
import torch.nn.functional as F
import numpy as np

from utils.cprint import cprint
from utils.meters import AverageMeter


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
        res -> dict('location': keypoints, ['angle': angles])
            keypoints -> list with shape [n, k, m, 3]
                All keypoints in heatmap with [x, y, score].
            [angles -> list with shape [n, k, m, 1]]
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
    kps = res['locations']
    if 'angles' in res:
        res_angles = res['angles']
    if 'sizes' in res:
        res_sizes = res['sizes']

    locations = [x.data for x in anns['locations']]
    labels = [x.data for x in anns['labels']]
    angles = [x.data for x in anns['angles']]
    sizes = [x.data for x in anns['sizes']]
    assert len(kps) == len(locations)

    offset = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    if 'angles' in res:
        angle_error = AverageMeter()
    if 'sizes' in res:
        size_error = AverageMeter()

    n_batch = len(kps)
    n_cls = len(kps[0])

    # for i -> batch, j -> class, i_d -> detection, i_t -> groundtruth
    for i in range(n_batch):
        for j in range(n_cls):
            dets = kps[i][j]
            tars = locations[i][labels[i] == j]
            # TODO: Whether raise except?
            tar_angle = angles[i][labels[i] == j]
            tar_size = sizes[i][labels[i] == j]

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
                        # dis
                        dis = math.sqrt(
                            calc_distance_square(dets[i_d], tars[i_t]))
                        offset.update(dis)
                        # angle
                        if 'angles' in res:
                            angle_off = abs(res_angles[i][j][i_d] - tar_angle[i_t])
                            angle_error.update(min(angle_off, 360-angle_off))
                            # cprint(f"res_angle: {res_angles[i][j][i_d]}    tar_angle: {tar_angle[i_t]}", level='debug')
                        # size
                        if 'sizes' in res:
                            size_error.update(abs(res_sizes[i][j][i_d] - tar_size[i_t]))
                            # cprint(f"res_size: {res_sizes[i][j][i_d]}    tar_angle: {tar_size[i_t]}", level='debug')

                        visit.append(i_t)
                        break
            precision.update(1, ok)
            precision.update(0, n_dets-ok)
            recall.update(1, ok)
            recall.update(0, n_tars-ok)

    out_dict = dict(dis=offset, precision=precision, recall=recall)
    if 'angles' in res:
        out_dict['angle_error'] = angle_error
    if 'sizes' in res:
        out_dict['size_error'] = size_error
    return out_dict


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
                        ['angle': np.array[n, 8, h, w]])
            The outputs from network.
        threshold -> Float
            The threshold for bg and fg.
        size -> Int
            The size of points used in nms.

    Output:
        out -> dict('location': keypoints, ['angle': angles])
            keypoints -> list with shape [n, k, m, 3]
                All keypoints in heatmap with [x, y, score].
            [angles -> list with shape [n, k, m, 1]]
    """
    out = {}

    # location
    keypoints = []

    heatmap = results['locations']
    batch, num_cls, h, w = heatmap.shape

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

            kps.append(res)
        keypoints.append(kps)
    out['locations'] = keypoints

    # angles
    if 'angles' in results:
        angle_map = results['angles']

    angles = []
    for i in range(batch):
        angle = []
        for j in range(num_cls):
            res = []
            for p in keypoints[i][j]:
                x, y = p[0], p[1]
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
                res.append(a)
            angle.append(res)
        angles.append(angle)

    out['angles'] = angles

    # size
    if 'sizes' in results:
        size_map = results['sizes']

    sizes = []
    for i in range(batch):
        size = []
        for j in range(num_cls):
            res = []
            for p in keypoints[i][j]:
                x, y = p[0], p[1]
                if 'sizes' in results:
                    s = size_map[i, 0, y, x]
                else:
                    s = -1
                res.append(s)
            size.append(res)
        sizes.append(size)

    out['sizes'] = sizes

    return out

