#!/usr/bin/env python
# coding=utf-8

import random

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

    return Pipline(config, cfg.MODEL.STRIDE, is_train)


class Pipline:
    """
    Process images and annotations before training or testing.
    """

    def __init__(self, cfg, stride, is_train):
        self.cfg = cfg
        self.stride = stride
        self.is_train = is_train

        # include [Resize, Flip, Lighting, Normalize]
        self.pipline = []
        self.pipline.append(Resize(self.cfg.SIZE,
                                   pad_color=self.cfg.MEAN,
                                   keep_ratio=True))
        if self.cfg.DO_FLIP:
            self.pipline.append(Flip(self.cfg.FLIP_PROB))
        if self.cfg.DO_SCALE:
            self.pipline.append(Scale(self.cfg.SCALE_RANGE, self.cfg.MEAN))
        if self.cfg.DO_ROTATE:
            self.pipline.append(Rotate(self.cfg.ROTATE_RANGE, self.cfg.MEAN))

    def __call__(self, results, num_cls):
        # image info
        trans_info = {}

        # ann
        ann = results['img_info']['ann']

        # opencv shape [h, w, c]
        img = cv2.imread(results['img_info']['filename'])
        # to rbg
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sample = dict(img=img, ann=ann)
        for func in self.pipline:
            sample = func(sample, trans_info)
        img = sample['img']
        ann = sample['ann']

        img_size = img.shape[:2]

        # to float
        img = img.astype(np.float32) / 255.0

        # normalize
        img = (img - self.cfg.MEAN) / self.cfg.STD

        img = img.transpose((2, 0, 1)).astype(np.float32)

        scale_factor = 1 if 'scale_factor' not in trans_info \
            else trans_info['scale_factor']
        target = self.get_gussian_targets(
            ann,
            img_size,
            self.stride,
            self.cfg.SIGMA * scale_factor,
            num_cls
        ) if self.is_train else []

        return dict(imgs=img,
                    targets=target,
                    trans_infos=trans_info,
                    anns=ann)

    def get_gussian_target(self, center, size, stride, sigma):
        """
        根据一个中心点，生成高斯热力图。

        Inputs:
            center: 一个中心点[x, y]

        Outputs:
            target: 这个中心点的热力图。
        """
        H, W = size
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

    def get_gussian_targets(self, ann, size, stride, sigma, num_cls):
        """
        Inputs:
            ...
            size: image size with list [h, w]
        Outputs:
            heatmaps: The heatmaps of labels with shape [n, num_cls, h, w].
        """
        H, W = size
        if H % stride != 0 or W % stride != 0:
            cprint(f"image size{size} is illegal", level='error')
            raise ValueError(f"image size is illegal")

        heatmaps = np.zeros((num_cls, H//stride, W//stride)).astype(np.float32)

        points = ann['locations']
        labels = ann['labels']
        num_points = points.shape[0]

        for i in range(num_points):
            label = labels[i]
            heatmap = self.get_gussian_target(points[i], size, stride, sigma)
            heatmaps[label] = np.maximum(heatmap, heatmaps[label])

        return heatmaps


class Resize:
    """
    Resize images and annotations(including location and size).

    Param:
        img_scale: The result images size in list [h, w].
        keep_ratio: Whether keep images ratio.

    Input:
        sample: A dict including target which will be processed.

    Output:
        sample: Result dict.
    """

    def __init__(self,
                 img_scale,
                 pad_color=[0, 0, 0],
                 keep_ratio=True):
        self.img_scale = img_scale
        # change range to 0-255
        self.pad_color = [int(x*255) for x in pad_color]
        self.keep_ratio = keep_ratio

    def __call__(self, sample, trans_info):
        img = sample['img']
        ann = sample['ann']

        trans_info['ori_size'] = img.shape[:2]

        ori_size = img.shape[:2]

        if self.keep_ratio:
            ratio = self._get_ratio(ori_size, self.img_scale)
            r_h, r_w = ratio, ratio
        else:
            r_h, r_w = self._get_ratio(ori_size, self.img_scale)

        trans_info['resize_ratio'] = [r_h, r_w]

        new_h = int(img.shape[0] * r_h + 0.5)
        new_w = int(img.shape[1] * r_w + 0.5)

        img = self._resize_img(img, (new_h, new_w), self.img_scale)
        ann = self._resize_ann(ann, (r_h, r_w))
        return dict(img=img, ann=ann)

    def _get_ratio(self, ori_size, target_size):
        h, w = ori_size
        target_h, target_w = target_size

        # 判断图片长宽的大小关系是否被改变
        if (h/w >= 1) != (target_h/target_w >= 1):
            cprint(f"ReizePipline | Image with shape [{h}, {w}] will "
                   f"resize to [{target_h}, {target_w}]",
                   level='error')
            raise ValueError(f"You can not resize images with this shape.")

        r_h = target_h / h
        r_w = target_w / w

        if self.keep_ratio:
            return min(r_h, r_w)
        else:
            return r_h, r_w

    def _resize_img(self, img, resize_shape, pad_shape):
        # tips: cv2.resize recieve [w, h]
        img = cv2.resize(img, (resize_shape[1], resize_shape[0]))

        bottom_pad = pad_shape[0] - resize_shape[0]
        right_pad = pad_shape[1] - resize_shape[1]

        img = cv2.copyMakeBorder(img, 0, bottom_pad, 0, right_pad,
                                 cv2.BORDER_CONSTANT, None, self.pad_color)

        return img

    def _resize_ann(self, ann, ratio):
        r_h, r_w = ratio

        # resize labels
        ann['locations'] = ann['locations'].astype(np.float) * [r_w, r_h]
        ann['sizes'] = ann['sizes'].astype(np.float) * max(r_h, r_w)

        return ann


class Flip:
    """
    Flip images and annotations(including location and direction).

    Param:
        prob: The probability 

    Input:
        sample: A dict including target which will be processed.

    Output:
        sample: Result dict.
    """

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample, trans_info):
        # TODO: support flip top-bottom
        img = sample['img']
        ann = sample['ann']

        if random.random() < self.prob:
            img = self._flip_img(img)
            ann = self._flip_ann(ann, img.shape[:2])
            trans_info['flip'] = True
        else:
            trans_info['flip'] = False

        return dict(img=img, ann=ann)

    def _flip_img(self, img):
        return img[:, ::-1, :]

    def _flip_ann(self, ann, size):
        _, w = size
        ann['locations'][:, 0] = -ann['locations'][:, 0] + w
        # TODO: support flip directions
        return ann


class Scale:
    """
    Scale images and annotations(including location and size)

    Param:
        scale_range: The scale range with list [start, end].
               `start` and `end` both less equal than 1.
        img_mean: Be used to fill empty pixels.

    Input:
        sample: A dict including target which will be processed.

    Output:
        sample: Result dict.
    """

    def __init__(self, scale_range, pad_color=[0, 0, 0]):
        if not isinstance(scale_range, list):
            cprint(f"scale_range is must be list, but get {type(scale_range)}",
                   level='error')
            raise TypeError(f"scale_range type error")
        elif len(scale_range) != 2:
            cprint(f"The length of scale_range is must be 2, but get {len(scale_range)}",
                   level='error')
            raise ValueError(f"scale_range value error")
        elif not all(0.1 <= x <= 1.0 for x in scale_range):
            cprint(f"scale_range must be in range [0.1, 1.0], but get {scale_range}",
                   level='error')
            raise ValueError(f"scale_range value error")

        self.scale_range = scale_range
        self.pad_color = [int(x*255) for x in pad_color]

    def __call__(self, sample, trans_info):
        img = sample['img']
        ann = sample['ann']

        scale_factor = random.uniform(*self.scale_range)

        img, shift = self._scale_img(img, scale_factor)
        ann = self._scale_ann(ann, scale_factor, shift)

        # save infos
        trans_info['scale_factor'] = scale_factor
        trans_info['scale_shift'] = shift

        return dict(img=img, ann=ann)

    def _scale_img(self, img, factor):
        h, w = img.shape[:2]
        new_h, new_w = int(h*factor), int(w*factor)

        img = cv2.resize(img, (new_w, new_h))

        top_pad = (h-new_h) // 2
        left_pad = (w-new_w) // 2
        bottom_pad = h - new_h - top_pad
        right_pad = w - new_w - left_pad

        img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad,
                                 cv2.BORDER_CONSTANT, None, self.pad_color)
        return img, (left_pad, top_pad)

    def _scale_ann(self, ann, factor, shift):
        """
        Input:
            ann: Dict 
            factor: Float
            shift: List with [shift_x, shift_y] means shift right and down.
        """
        shift_x, shift_y = shift

        ann['locations'] = \
            ann['locations'].astype(np.float) * factor + [shift_x, shift_y]
        ann['sizes'] = ann['sizes'].astype(np.float) * factor
        return ann


class Rotate:
    """
    Rotate images and annotations(including location and direction)

    Param:
        rotate_range: The rotate range with List [start, end].
            `start` and `end` in range [-180, 180]
    Input:
        sample: A dict including target which will be processed.

    Output:
        sample: Result dict.
    """

    def __init__(self, rotate_range, pad_color=[0, 0, 0]):
        if not isinstance(rotate_range, list):
            cprint(f"rotate_range is must be list, but get {type(rotate_range)}",
                   level='error')
            raise TypeError(f"rotate_range type error")
        elif len(rotate_range) != 2:
            cprint(f"The length of rotate_range is must be 2, but get {len(rotate_range)}",
                   level='error')
            raise ValueError(f"rotate_range value error")
        elif not all(-180 <= x <= 180 for x in rotate_range):
            cprint(f"rotate_range must be in range [-180, 180], but get {rotate_range}",
                   level='error')
            raise ValueError(f"rotate_range value error")

        self.rotate_range = rotate_range
        self.pad_color = [int(x*255) for x in pad_color]

    def __call__(self, sample, trans_info):
        img = sample['img']
        ann = sample['ann']

        rotate_angle = random.randrange(*self.rotate_range)
        trans_info['rotate_angle'] = rotate_angle

        # M.shape: [2, 3]
        M = self._get_affine_matrix(img.shape[:2], rotate_angle)

        img = self._rotate_img(img, M)
        ann = self._rotate_ann(ann, M)

        return dict(img=img, ann=ann)

    def _get_affine_matrix(self, size, rotate_angle):
        h, w = size
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, rotate_angle, scale=1.0)
        return M
        
    def _rotate_img(self, img, M):
        # warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])
        # dsize is [w, h]
        img = cv2.warpAffine(img, M, img.shape[:2][::-1],
                             borderValue=self.pad_color)
        return img

    def _rotate_ann(self, ann, M):
        ann['locations'] = np.insert(ann['locations'], 2, 1, axis=1).T # shape [3, n]
        ann['locations'] = np.matmul(M, ann['locations']).T # [2, 3] * [3, n]
        # TODO: rotate directions
        return ann

