#!/usr/bin/env python
# coding=utf-8

import random
import math

import torchvision.transforms as T
from torchvision.transforms import ColorJitter, ToPILImage
import cv2
import torch
import numpy as np
from utils.cprint import cprint


def resume_imgs(imgs, mean, std):
    """
    Resume images from tensor to numpy.

    Inputs:
        imgs: Tensor in range(0.0, 1.0) with shape [n, c, h, w].
        mean: Mean of images with [R, G, B].
        std: Std of images with [R, G, B].

    Outputs:
        imgs: Numpy in range(0, 255) with shape [n, h, w, c].
    """
    imgs = imgs.numpy()
    imgs = imgs.transpose(0, 2, 3, 1) # to [n, h, w, c]
    imgs = (imgs*std + mean)*255
    imgs = imgs.astype(np.uint8)
    return imgs


def build_transforms(cfg, is_train):
    if is_train:
        config = cfg.TRAIN
    else:
        config = cfg.TEST

    return Pipline(config, cfg.MODEL, is_train)


class Pipline:
    """
    Process images and annotations before training or testing.
    """

    def __init__(self, cfg, model_cfg, is_train):
        self.cfg = cfg
        self.stride = model_cfg.STRIDE
        self.train_size = model_cfg.SIZE
        self.train_angle = model_cfg.ANGLE
        self.is_train = is_train

        # include [Resize, Flip, Lighting, Normalize]
        self.pipline = []
        self.pipline.append(Resize(self.cfg.IMG_SIZE,
                                   pad_color=self.cfg.MEAN,
                                   keep_ratio=True))
        if self.cfg.DO_FLIP:
            self.pipline.append(Flip(self.cfg.FLIP_PROB))
        if self.cfg.DO_SCALE:
            self.pipline.append(Scale(self.cfg.SCALE_RANGE, self.cfg.MEAN))
        if self.cfg.DO_ROTATE:
            self.pipline.append(Rotate(self.cfg.ROTATE_RANGE, self.cfg.MEAN))
        if self.cfg.DO_ALBU:
            self.pipline.append(Albu(self.cfg.ALBU_BRIGHTNESS,
                                     self.cfg.ALBU_CONTRAST,
                                     self.cfg.ALBU_SATURATION,
                                     self.cfg.ALBU_HUE))

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
        # remove points which out of image
        if 'scale_factor' in trans_info \
                and trans_info['scale_factor'] > 1.0:
            ann = self.remove_out_points(ann, img_size)

        # to float
        img = img.astype(np.float32) / 255.0

        # normalize
        img = (img - self.cfg.MEAN) / self.cfg.STD

        img = img.transpose((2, 0, 1)).astype(np.float32)

        targets = {}
        # heatmaps [n, k, h, w]
        scale_factor = 1 if 'scale_factor' not in trans_info \
            else trans_info['scale_factor']
        targets['locations'] = self.get_gussian_targets(
            ann,
            img_size,
            self.stride,
            self.cfg.SIGMA * scale_factor,
            num_cls
        ) if self.is_train else []

        # angle maps [n, 2, h, w] (channel=2 means sin and cos value)
        if self.is_train and self.train_angle:
            targets['angles'] = self.get_angle_targets(
                ann,
                img_size,
                self.stride,
                self.cfg.SIGMA * scale_factor,
            )

        return dict(imgs=img,
                    targets=targets,
                    trans_infos=trans_info,
                    anns=ann)

    def remove_out_points(self, ann, img_size):
        idx = np.where((ann['locations'].data[:, 1]<=img_size[0]-1)& \
                       (ann['locations'].data[:,0]<=img_size[1]))

        ann['locations'].data = ann['locations'].data[idx]
        ann['sizes'].data = ann['sizes'].data[idx]
        ann['angles'].data = ann['angles'].data[idx]
        ann['labels'].data = ann['labels'].data[idx]
        return ann


    def get_angle_target(self, center, angle, size, stride, sigma):
        """
        根据一个中心点和一个角度，生成角度图。
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
        exponent = np.exp(-exponent)

        angle_map = np.zeros((map_h, map_w, 2)) - 1  # [h, w, 2]
        values = [math.sin(angle/180*math.pi), math.cos(angle/180*math.pi)]
        angle_map[exponent>0.01] = values
        angle_map = angle_map.transpose(2, 0, 1)  # [2, h, w]
        return angle_map


    def get_angle_targets(self, ann, size, stride, sigma):
        """
        根据中心点生成角度地图，-1表示忽略的点。

        Outputs:
            angle_maps: shape [n, 2, h, w].
        """
        H, W = size
        points = ann['locations'].data
        angles = ann['angles'].data
        num_points = points.shape[0]

        angle_maps = np.zeros((2, H//stride, W//stride)).astype(np.float32) - 1

        for i in range(num_points):
            angle = angles[i]
            angle_map = self.get_angle_target(points[i], angle, size, stride, sigma)
            angle_maps = np.maximum(angle_maps, angle_map)

        return angle_maps


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

        points = ann['locations'].data
        labels = ann['labels'].data
        num_points = points.shape[0]

        for i in range(num_points):
            label = labels.data[i]
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
        ann['locations'].data = ann['locations'].data.astype(np.float) * [r_w, r_h]
        ann['sizes'].data = ann['sizes'].data.astype(np.float) * max(r_h, r_w)

        return ann


class Flip:
    """
    Flip images and annotations(including location and angle).

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
        ann['locations'].data[:, 0] = -ann['locations'].data[:, 0] + w

        ann['angles'].data = np.remainder(-ann['angles'].data + 180, 360)

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
        elif not all(0.1 <= x <= 2.0 for x in scale_range):
            cprint(f"scale_range must be in range [0.1, 2.0], but get {scale_range}",
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

        return dict(img=img, ann=ann)

    def _scale_img(self, img, factor):
        h, w = img.shape[:2]
        new_h, new_w = int(h*factor), int(w*factor)

        img = cv2.resize(img, (new_w, new_h))

        if factor <= 1.0:
            top_pad = (h-new_h) // 2
            left_pad = (w-new_w) // 2
            bottom_pad = h - new_h - top_pad
            right_pad = w - new_w - left_pad

            img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad,
                                     cv2.BORDER_CONSTANT, None, self.pad_color)
            return img, (left_pad, top_pad)
        else:
            shift_x = (new_w - w) // 2
            shift_y = (new_h - h) // 2

            img = img[shift_y: shift_y+h, shift_x: shift_x+w, :]
            return img, (shift_x, shift_y)

    def _scale_ann(self, ann, factor, shift):
        """
        Input:
            ann: Dict 
            factor: Float
            shift: List with [shift_x, shift_y] means shift right and down.
        """
        shift_x, shift_y = shift

        if factor <= 1.0:
            ann['locations'].data = \
                ann['locations'].data.astype(np.float) * factor + [shift_x, shift_y]
            ann['sizes'].data = ann['sizes'].data.astype(np.float) * factor
            return ann
        else:
            ann['locations'].data = \
                ann['locations'].data.astype(np.float) * factor - [shift_x, shift_y]
            ann['sizes'].data = ann['sizes'].data.astype(np.float) * factor
            return ann


class Rotate:
    """
    Rotate images and annotations(including location and angle)

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
        ann = self._rotate_ann(ann, M, rotate_angle)

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

    def _rotate_ann(self, ann, M, rotate_angle):
        ann['locations'].data = \
            np.insert(ann['locations'].data, 2, 1, axis=1).T  # shape [3, n]
        ann['locations'].data = \
            np.matmul(M, ann['locations'].data).T  # [2, 3] * [3, n]

        ann['angles'].data = np.remainder(ann['angles'].data + rotate_angle, 360)
        return ann


class Albu:
    """
    Transform Image with color including brightness, contrast, saturation and hue.

    Input:
        sample: A dict including target which will be processed.

    Output:
        sample: Result dict.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.to_tensor = ToPILImage()
        self.albu = ColorJitter(brightness=brightness,
                                contrast=contrast,
                                saturation=saturation,
                                hue=hue)

    def __call__(self, sample, tran_info):
        img = sample['img']
        ann = sample['ann']

        img = self.albu(self.to_tensor(img))
        img = np.asarray(img)
        return dict(img=img, ann=ann)
