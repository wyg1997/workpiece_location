#!/usr/bin/env python
# coding=utf-8

import os
import os.path as osp
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
from utils.cprint import cprint
from datasets.data_container import DataContainer


class ImageDataset:

    """Custom dataset for location.
    Folder stucture:
        --| data_root
        ----| souce
        ----| label
    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'locations': <np.ndarray> (n, 2),
                'labels': <np.ndarray> (n, ),
                'sizes': <np.ndarray> (n, )
                'directions': <np.ndarray> (n, )
            }
        },
        ...
    ]
    The `ann` field is optional for testing.
    """

    all_img_type = {'png', 'jpg', 'bmp'}

    CLASSES = []
    cat2label = dict()

    def __init__(self,
                 cfg,
                 pipeline,
                 test_mode=False,
                 CLASSES=None):

        self.img_type = cfg.IMG_TYPE
        assert self.img_type in self.all_img_type, f"img_type must be one of {self.all_img_type}"

        self.data_root = osp.abspath(cfg.DATA_ROOT)
        self.img_index = cfg.IMG_INDEX
        self.test_mode = test_mode

        # classes
        if CLASSES is not None:
            self.CLASSES = CLASSES
            for i, cls in enumerate(CLASSES):
                self.cat2label[cls] = i+1

        # load annotations (and proposals)
        self.img_infos = self.load_annotations(self.data_root)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # processing pipeline
        self.pipeline = pipeline

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, data_root):
        img_infos = []
        if len(self.img_index) == 0:
            self.img_ids = self._get_img_index(data_root)
        else:
            self.img_ids = self.img_index
        for img_id in self.img_ids:
            filename = osp.join(data_root, f"source/{img_id}.{self.img_type}")
            xml_path = osp.join(data_root, f"label/{img_id}.xml")
            gt = self._load_ground_truth(xml_path)
            img_infos.append(
                dict(filename=filename, **gt))
        return img_infos

    def _load_ground_truth(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        locations = []
        labels = []
        sizes = []
        directions = []

        # size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # gts
        for point in root.findall('point'):
            name = point.find('name').text

            # check label
            if name not in self.CLASSES:
                self.CLASSES.append(name)
                self.cat2label[name] = len(self.CLASSES)
            label = self.cat2label[name]

            loc = point.find('location')
            location = [
                int(float(loc.find('x').text)),
                int(float(loc.find('y').text))
            ]

            r = int(float(point.find('size').text))
            direction = float(point.find('direction').text)

            locations.append(location)
            labels.append(label)
            sizes.append(r)
            directions.append(direction)

        if not locations:
            locations = np.zeros((0, 2))
            labels = np.zeros((0, ))
            sizes = np.zeros((0, ))
            directions = np.zeros((0, ))
        else:
            locations = np.array(locations, ndmin=2) - 1
            labels = np.array(labels)
            sizes = np.array(sizes)
            directions = np.array(directions)

        ann = dict(
            locations = DataContainer(locations.astype(np.int)),
            labels = DataContainer(labels.astype(np.int)),
            sizes = DataContainer(sizes.astype(np.int)),
            directions = DataContainer(directions.astype(np.float))
        )
        return dict(width=width, height=height, ann=ann)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['img_prefix'] = osp.join(self.data_root, 'source')
        results['proposal_file'] = osp.join(self.data_root, 'label')

    def _get_img_index(self, data_root):
        """Read ids from data root path"""
        ids = []
        source_path = osp.join(data_root, 'source')
        label_path = osp.join(data_root, 'label')
        assert osp.exists(source_path) and osp.exists(label_path), \
                "Data path not exist, please check your path!"
        files = os.listdir(source_path)
        for f in files:
            id, file_type = osp.splitext(f)
            if file_type[1:] != self.img_type:
                continue
            if not osp.exists(osp.join(label_path, id+'.xml')):
                cprint(f"AIDIDataset | {id}.xml is not exist!", level='warn')
                continue
            ids.append(id)
        return ids

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        return self.prepare_img(idx)

    def prepare_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=deepcopy(img_info))
        self.pre_pipeline(results)
        return self.pipeline(results, len(self.CLASSES)+1)
