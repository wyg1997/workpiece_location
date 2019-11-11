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
from models.FCN import FCNs, VGGNet
from utils.meters import AverageMeter
from utils.cprint import cprint
from utils.visualize import visualize


class Trainer:
    def __init__(self, cfg, logger, vis, work_dir):
        self.cfg, self.logger, self.vis, self.work_dir = cfg, logger, vis, work_dir
        
        # dataloader
        self.train_dataloader, self.val_dataloader, self.classes = get_dataloader(cfg)
        self.num_cls = len(self.classes) + 1
        logger.info(f"classes: {self.classes}")

        # model
        vgg_model = VGGNet(requires_grad=True, show_params=False)

        self.model = FCNs(pretrained_net=vgg_model, n_class=len(self.classes)+1)

        self.logger.info(f"model: \n{self.model}")

        # criterion
        self.criterion = nn.MSELoss()

        # TODO: support more optimizer
        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.7)

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
            loss = self.criterion(outputs, targets)
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

            # see train data
            if i == 0:
                vis_images = visualize(ori_imgs, ori_targets,
                          mean=self.cfg.TRAIN.MEAN, std=self.cfg.TRAIN.STD)
                self.vis.images(vis_images)


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

