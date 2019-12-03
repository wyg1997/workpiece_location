#!/usr/bin/env python
# coding=utf-8

import os.path as osp
import time

import torch
import numpy as np

from datasets.build import get_dataloader
from utils.meters import AverageMeter
from utils.cprint import cprint
from solver.build import make_optimizer, make_lr_scheduler, make_loss_function
from tools.visualize import visualize
from tools.kps_tools import get_kps_from_heatmap, eval_key_points
from models.build import build_model
from .tester import Tester


class Trainer:
    def __init__(self, cfg, logger, vis, work_dir):
        self.cfg, self.logger, self.vis, self.work_dir = cfg, logger, vis, work_dir

        # dataloader
        self.train_dataloader, self.classes = \
            get_dataloader(cfg, kind='train', CLASSES=cfg.MODEL.CLASSES)
        self.num_cls = len(self.classes)
        self.logger.info(f"classes: {self.classes}")

        # model
        # fcn
        self.model = build_model(cfg.MODEL.BACKBONE, self.num_cls, cfg.MODEL.PRETRAIN)

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
        self.criterion = self.criterion.cuda()

        self.running_loss = AverageMeter()

    def on_epoch_begin(self):
        self.current_epoch += 1
        self.time0 = time.time()
        self.running_loss.reset()

    def on_epoch_end(self):
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
            self.tester.test(threshold=0.5, show=True)
        print()

    def training_epoch(self):
        for i, data in enumerate(self.train_dataloader):
            start_time = time.time()
            ori_imgs = data['imgs']
            ori_targets = data['targets']

            imgs = ori_imgs.cuda()
            targets = ori_targets.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(imgs)

            # get loss
            loss = self.criterion(outputs, targets)

            # backward step
            loss.backward()
            self.optimizer.step()

            # cost time
            cost_time = time.time() - start_time

            # precisions of results
            results = outputs.cpu().detach()
            kps = get_kps_from_heatmap(results,
                                       self.cfg.MODEL.STRIDE,
                                       threshold=0.5,
                                       size=self.cfg.TRAIN.SIZE)
            dis, p, r = eval_key_points(kps, data['anns'], size=40)

            loss = loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']

            # write log
            if i % self.cfg.SOLVER.LOG_INTERVAL == 0:
                self.logger.info(f"Epoch: {self.current_epoch} | "
                                 f"Iter: {i+1}/{len(self.train_dataloader)} | "
                                 f"lr: {current_lr:.2e} | "
                                 f"loss: {loss:.4f} | "
                                 f"dis_loss: {dis:.2f} | "
                                 f"precision: {p:.2%} | "
                                 f"recall: {r:.2%} | "
                                 f"time: {cost_time*1000:.0f}ms")

            # update iter
            self.running_loss.update(loss)
            self.global_step += 1

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
            self.vis.line(Y=np.array([dis]),
                          X=np.array([self.global_step]),
                          win='dis_loss',
                          update=None if self.global_step == 1 else 'append',
                          opts=dict(title='train_dis_loss'))

            # see train data
            if i == 0:
                vis_images = visualize(ori_imgs, ori_targets,
                                       stride=self.cfg.MODEL.STRIDE,
                                       mean=self.cfg.TRAIN.MEAN,
                                       std=self.cfg.TRAIN.STD)
                self.vis.images(vis_images, win='label',
                                opts=dict(title='label'))

            # see train results
            if i == 0:
                vis_images = visualize(ori_imgs, results,
                                       stride=self.cfg.MODEL.STRIDE,
                                       mean=self.cfg.TRAIN.MEAN,
                                       std=self.cfg.TRAIN.STD,
                                       alpha=0.5)
                self.vis.images(vis_images, win='train_results',
                                opts=dict(title='train_results'))

            # # see 2D heatmap
            # if i == 0:
            #     for cls_idx in range(self.num_cls):
            #         self.vis.surf(results[0][cls_idx],
            #                       win=f"heatmap_cls_{cls_idx}",
            #                       opts=dict(title=f"heatmap_cls_{cls_idx}"))
            #     for cls_idx in range(self.num_cls):
            #         self.vis.surf(ori_targets[0][cls_idx],
            #                       win=f"target_heatmap_cls_{cls_idx}",
            #                       opts=dict(title=f"target_heatmap_cls_{cls_idx}"))
            #     # input('press enter to continue...')

    def train(self):
        self.on_train_begin()

        for epoch in range(self.max_epochs):
            self.on_epoch_begin()

            self.training_epoch()

            self.on_epoch_end()

            if (epoch+1) % self.cfg.SOLVER.CHECKPOINT == 0:
                self.save_checkpoints()

            torch.cuda.empty_cache()

    def save_checkpoints(self):
        torch.save(self.model.state_dict(),
                   osp.join(self.work_dir, str(self.current_epoch))+'.pth')
