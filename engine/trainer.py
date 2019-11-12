#!/usr/bin/env python
# coding=utf-8

import os
import sys
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.autograd import Variable

from datasets.build import get_dataloader
from utils.meters import AverageMeter
from utils.cprint import cprint
from utils.visualize import visualize


class Trainer:
    def __init__(self, cfg, logger, vis, work_dir):
        self.cfg, self.logger, self.vis, self.work_dir = cfg, logger, vis, work_dir
        
        # dataloader
        self.train_dataloader, self.val_dataloader, self.classes = get_dataloader(cfg)
        self.num_cls = len(self.classes) + 1
        self.logger.info(f"classes: {self.classes}")

        # model
        # hg
        from models.hg import hg
        self.model = hg(num_stacks=4,
                        num_blocks=4,
                        num_classes=5,
                        resnet_layers=50)

        # self.logger.info(f"model: \n{self.model}")

        # criterion
        self.criterion = nn.MSELoss()

        # TODO: support more optimizer
        # optimizer
        self.base_lr = 1e-4
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                   lr=self.base_lr,
                                   momentum=0,
                                   weight_decay=0)

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
        # opt
        # TODO: put it in config file
        if self.current_epoch in [24, 48, 56, 72]:
            self.base_lr *= 0.5
            self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.base_lr,
                                       momentum=0,
                                       weight_decay=0)

        self.current_epoch += 1
        self.time0 = time.time()
        self.running_loss.reset()

    def on_epoch_end(self):
        elapsed = time.time() - self.time0
        mins = int(elapsed) // 60
        seconds = int(elapsed - mins*60)

        print()
        self.logger.info(f"Epoch {self.current_epoch} | "
                         f"time {mins:d}min:{seconds:d}s | "
                         f"loss {self.running_loss.avg:.3f}")
        print()

    def training_epoch(self):
        for i, (ori_imgs, ori_targets) in enumerate(self.train_dataloader):
            start_time = time.time()
            imgs = Variable(ori_imgs).cuda()
            targets = Variable(ori_targets).cuda()

            self.optimizer.zero_grad()
            outputs = self.model(imgs)

            loss = self.criterion(outputs[0], targets)
            for j in range(1, len(outputs)):
                loss += self.criterion(outputs[j], targets)

            loss.backward()
            self.optimizer.step()

            # cost time
            cost_time = time.time() - start_time

            loss = loss.item()
            self.logger.info(f"Epoch {self.current_epoch} | "
                             f"Iter {i+1}/{len(self.train_dataloader)} | "
                             f"loss {loss:.4f} | "
                             f"time {cost_time*1000:.0f}ms")

            self.running_loss.update(loss)
            self.global_step += 1
            self.vis.line(Y=np.array([loss]),
                          X=np.array([self.global_step]),
                          win='train_loss',
                          update=None if self.global_step==0 else 'append')

            # # see train data
            # if i == 0:
            #     vis_images = visualize(ori_imgs, ori_targets,
            #                            stride=self.cfg.TRAIN.STRIDE,
            #                            mean=self.cfg.TRAIN.MEAN,
            #                            std=self.cfg.TRAIN.STD)
            #     self.vis.images(vis_images, win='label_iter0',
            #                     opts=dict(title='train iter0'))

            # see train results
            if i == 0:
                results = outputs[-1].cpu().detach()
                results[results<0.3] = 0
                vis_images = visualize(ori_imgs, results,
                                       stride=self.cfg.TRAIN.STRIDE,
                                       mean=self.cfg.TRAIN.MEAN,
                                       std=self.cfg.TRAIN.STD,
                                       alpha=0.8)
                self.vis.images(vis_images, win='test_iter0',
                                opts=dict(title='test iter0'))


    def train(self):
        self.on_train_begin()

        for epoch in range(self.max_epochs):
            self.on_epoch_begin()

            self.training_epoch()

            self.on_epoch_end()
            self.save_checkpoints()

            torch.cuda.empty_cache()

    def save_checkpoints(self):
        torch.save(self.model.state_dict(),
                   osp.join(self.work_dir, str(self.current_epoch))+'.pth')

