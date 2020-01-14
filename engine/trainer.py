#!/usr/bin/env python
# coding=utf-8

import os
import os.path as osp
import time

import torch
import numpy as np
from prefetch_generator import BackgroundGenerator

from datasets.build import get_dataloader
from datasets.transforms import resume_imgs
from utils.meters import AverageMeter
from utils.cprint import cprint
from solver.build import make_optimizer, make_lr_scheduler, make_loss_function
from tools.visualize import vis_heatmaps, vis_anns, vis_results
from tools.kps_tools import get_kps_from_heatmap, eval_key_points, resize_heatmaps
from models.build import build_model
from .tester import Tester
from solver.loss_functions import angle_loss_func, size_loss_func


class Trainer:
    def __init__(self, cfg, logger, vis, work_dir):
        self.cfg, self.logger, self.vis, self.work_dir = cfg, logger, vis, work_dir

        # dataloader
        self.dataset, self.train_dataloader, self.classes = \
            get_dataloader(cfg, kind='train', CLASSES=cfg.MODEL.CLASSES)
        self.num_cls = len(self.classes)
        self.logger.info(f"classes: {self.classes}")

        # model
        # task
        self.task = ['locations']
        if self.cfg.MODEL.ANGLE:
            self.task.append('angles')
        if self.cfg.MODEL.SIZE:
            self.task.append('sizes')
        # fcn
        self.model = build_model(cfg.MODEL, self.num_cls)
        if self.cfg.MODEL.CHECKPOINT:
            self.logger.info(f"load pretrain from `{self.cfg.MODEL.CHECKPOINT}`")
            self.model.load_state_dict(torch.load(self.cfg.MODEL.CHECKPOINT)['checkpoint'])
        else:
            cprint(f"no weights loaded.")

        # Tester
        self.tester = Tester(cfg, logger, vis, work_dir, self.model, self.classes)

        # criterion
        self.criterion = make_loss_function(self.cfg.SOLVER.LOSS)

        # optimizer
        self.optimizer = make_optimizer(cfg.SOLVER, self.model)
        # lr scheduler
        self.lr_scheduler = make_lr_scheduler(cfg.SOLVER, self.optimizer)

        self._construct()

    def _construct(self):
        self.global_step = 0
        self.current_epoch = 0
        self.max_epochs = self.cfg.SOLVER.EPOCHS
        self.log_interval = self.cfg.SOLVER.LOG_INTERVAL

    def on_train_begin(self):
        self.model = self.model.cuda()
        self.model.train()

        self.running_loss = AverageMeter()

    def on_epoch_begin(self):
        self.current_epoch += 1
        self.time0 = time.time()
        self.running_loss.reset()

    def on_epoch_end(self):
        # save model
        if self.current_epoch % self.cfg.SOLVER.CHECKPOINT == 0:
            self.save_checkpoints()

        elapsed = time.time() - self.time0
        mins = int(elapsed) // 60
        seconds = int(elapsed - mins*60)
        # step lr
        self.lr_scheduler.step()

        print()
        self.logger.info(f"Epoch: {self.current_epoch} | "
                         f"time: {mins:d}min:{seconds:d}s | "
                         f"loss: {self.running_loss.avg:.5f}")
        # test
        if self.current_epoch % self.cfg.SOLVER.EVAL_EPOCH == 0:
            self.tester.test(threshold=0.5, show=self.cfg.VISDOM.SHOW_TEST_OUT)
        print()

    def on_train_end(self):
        self.save_checkpoints(file_name='latest')

    def training_epoch(self):
        """
        Train an epoch. Forward and backward in each iter.

        It will do
            1. Load datas and forward.
            2. Calculate loss and backward.
            3. Collate and calculate results(locations and angles).
            4. Eval results and get dis_error, precision, recall and angle_error.
            5. Print to logger.
            6. Show results to visdom.
        in each iter.
        """
        for i, data in enumerate(BackgroundGenerator(self.train_dataloader)):
            start_time = time.time()
            # 1. Load datas and forward.
            ori_imgs = data['imgs']
            ori_targets = data['targets']

            # move to gpu
            imgs = ori_imgs.cuda()
            targets = {}
            for key in self.task:
                targets[key] = ori_targets[key].cuda()

            self.optimizer.zero_grad()
            outputs = self.model(imgs)

            # 2. Calculate loss and backward.
            # get loss
            # location
            location_loss = self.criterion(outputs['locations'][0],
                                           targets['locations'])
            for out_idx in range(1, len(outputs['locations'])):
                location_loss += self.criterion(outputs['locations'][out_idx],
                                                targets['locations'])
            loss = location_loss

            # angle
            if 'angles' in self.task:
                angle_weight = 1.0
                mask = (outputs['locations'][0].max(axis=1)[0] >= 0.5).unsqueeze(1).repeat(1, 8, 1, 1)
                mask = mask & (targets['angles'] != -1)

                angle_loss = angle_loss_func(outputs['angles'][0],
                                             targets['angles'],
                                             mask) * angle_weight
                for out_idx in range(1, len(outputs['angles'])):
                    mask = (outputs['locations'][out_idx].max(axis=1)[0] >= 0.5).unsqueeze(1).repeat(1, 8, 1, 1)
                    mask = mask & (targets['angles'] != -1)
                    angle_loss += angle_loss_func(outputs['angles'][out_idx],
                                                 targets['angles'],
                                                 mask) * angle_weight
                loss += angle_loss
            else:
                angle_loss = -1

            # size
            if 'sizes' in self.task:
                size_weight = 1.0
                mask = (outputs['locations'][0].max(axis=1)[0] >= 0.5).unsqueeze(1)
                mask = mask & (targets['sizes'] != -1)

                size_loss = size_loss_func(outputs['sizes'][0],
                                           targets['sizes'],
                                           mask) * size_weight
                for out_idx in range(1, len(outputs['sizes'])):
                    mask = (outputs['locations'][out_idx].max(axis=1)[0] >= 0.5).unsqueeze(1)
                    mask = mask & (targets['sizes'] != -1)
                    size_loss += size_loss_func(outputs['sizes'][out_idx],
                                                targets['sizes'],
                                                mask) * size_weight
                loss += size_loss
            else:
                size_loss = -1

            # backward step
            loss.backward()
            self.optimizer.step()

            # cost time
            cost_time = time.time() - start_time

            # 3. Collate and calculate results(locations and angles).
            # precisions of results
            results = {}
            for key in self.task:
                results[key] = resize_heatmaps(outputs[key][-1], self.cfg.MODEL.STRIDE)

            kps = get_kps_from_heatmap(results,
                                       threshold=0.5,
                                       size=self.cfg.TRAIN.SIZE)

            # 4. Eval results and get dis_error, precision, recall and angle_error.
            eval_res = eval_key_points(kps, data['anns'], size=40)
            dis = eval_res['dis']
            p = eval_res['precision']
            r = eval_res['recall']
            # angle error
            angle_error = eval_res['angle_error']
            if angle_error.count == 0:
                angle_error.update(-1)
            # size error
            size_error = eval_res['size_error']
            if size_error.count == 0:
                size_error.update(-1)

            loss = loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 5. Print to logger.
            # write log
            if (i+1) % self.cfg.SOLVER.LOG_INTERVAL == 0:
                self.logger.info(
                    f"Epoch: {self.current_epoch} | "
                    f"Iter: {i+1}/{len(self.train_dataloader)} | "
                    f"lr: {current_lr:.2e} | "
                    f"location_loss: {location_loss:.4f} | "
                    f"angle_loss: {angle_loss:.4f} | "
                    f"size_loss: {size_loss:.4f} | "
                    f"dis_offset: {dis.avg:.2f} | "
                    f"angle_error: {angle_error.avg:.4f} | "
                    f"size_error: {size_error.avg:.2f} | "
                    f"precision: {p.avg:.2%} | "
                    f"recall: {r.avg:.2%} | "
                    f"time: {cost_time*1000:.0f}ms")

            # update iter
            self.running_loss.update(loss)
            self.global_step += 1

            # 6. Show results to visdom.
            # show results and labels
            self.vis.text(f"{kps}", win='results')
            self.vis.text(f"{data['anns']}", win='anns')

            # loss line
            self.vis.line(Y=np.array([loss]),
                          X=np.array([self.global_step]),
                          win='train_loss',
                          update=None if self.global_step == 1 else 'append',
                          opts=dict(title='train_loss'))
            # lr line
            self.vis.line(Y=np.array([current_lr]),
                          X=np.array([self.global_step]),
                          win='lr',
                          update=None if self.global_step == 1 else 'append',
                          opts=dict(title='train_lr'))

            # dis line
            self.vis.line(Y=np.array([dis.avg]),
                          X=np.array([self.global_step]),
                          win='dis_error',
                          update=None if self.global_step == 1 else 'append',
                          opts=dict(title='train_dis_error'))

            # angle line
            self.vis.line(Y=np.array([angle_error.avg]),
                          X=np.array([self.global_step]),
                          win='angle_error',
                          update=None if self.global_step == 1 else 'append',
                          opts=dict(title='train_angle_error'))

            # size line
            self.vis.line(Y=np.array([size_error.avg]),
                          X=np.array([self.global_step]),
                          win='size_error',
                          update=None if self.global_step == 1 else 'append',
                          opts=dict(title='train_size_error'))

            # resume origin images
            if i == 0:
                ori_imgs = resume_imgs(ori_imgs,
                                       self.cfg.TRAIN.MEAN,
                                       self.cfg.TRAIN.STD)

            # see labels
            if self.cfg.VISDOM.SHOW_LABEL and i == 0:
                label_img = vis_anns(np.copy(ori_imgs), data['anns'],
                                     self.classes,
                                     self.cfg.VISDOM.SHOW_INFO)
                self.vis.images(label_img, win='label_image',
                                opts=dict(title='label_image'))

            # see all results(location and angle)
            if self.cfg.VISDOM.SHOW_TRAIN_OUT and i == 0:
                res_img = vis_results(np.copy(ori_imgs), kps,
                                      self.classes,
                                      self.cfg.VISDOM.SHOW_INFO)
                self.vis.images(res_img, win='result_image',
                                opts=dict(title='result_image'))

            # see label heatmaps
            if self.cfg.VISDOM.SHOW_LABEL_HEATMAP and i == 0:
                targets = resize_heatmaps(targets['locations'], self.cfg.MODEL.STRIDE)
                vis_images = vis_heatmaps(np.copy(ori_imgs), targets, alpha=0.5)
                self.vis.images(vis_images, win='label',
                                opts=dict(title='label'))

            # see train heatmap results
            if self.cfg.VISDOM.SHOW_TRAIN_HEATMAP and i == 0:
                vis_images = vis_heatmaps(np.copy(ori_imgs), results['locations'], alpha=0.5)
                self.vis.images(vis_images, win='train_heatmap',
                                opts=dict(title='train_heatmap'))


    def train(self):
        self.on_train_begin()

        for epoch in range(self.max_epochs):
            self.on_epoch_begin()

            self.training_epoch()

            self.on_epoch_end()

            # torch.cuda.empty_cache()

        self.on_train_end()

    def save_checkpoints(self, file_name=None):
        save_dir = osp.join(self.work_dir, 'checkpoints/')
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        if file_name is not None:
            file_name = file_name + '.pth'
        else:
            file_name = str(self.current_epoch) + '.pth'
        torch.save(dict(checkpoint=self.model.state_dict(),
                        classes=self.classes),
                   osp.join(save_dir, file_name))
