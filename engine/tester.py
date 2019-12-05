#!/usr/bin/env python
# coding=utf-8

import torch
import numpy as np

from datasets.build import get_dataloader
from tools.kps_tools import get_kps_from_heatmap, eval_key_points
from utils.meters import AverageMeter
from utils.cprint import cprint
from tools.visualize import visualize


class Tester:
    def __init__(self, cfg, logger, vis, work_dir, model, classes):
        self.cfg, self.logger, self.vis, self.work_dir = cfg, logger, vis, work_dir
        self.model, self.classes = model, classes
        self.num_cls = len(classes) + 1

        self.test_dataloader = get_dataloader(cfg, kind='test', CLASSES=classes)

        self.test_cnt = 0

    def test(self, threshold=0.5, show=False):
        self.model.eval()

        eval_dis = AverageMeter()
        eval_p = AverageMeter()
        eval_r = AverageMeter()

        for i, data in enumerate(self.test_dataloader):
            ori_imgs = data['imgs']

            imgs = ori_imgs.cuda()
            with torch.no_grad():
                outputs = self.model(imgs)
            if isinstance(outputs, list):
                results = outputs[-1].cpu().detach()
            else:
                results = outputs.cpu().detach()

            # vis results
            if show:
                vis_img = visualize(ori_imgs, results,
                                    stride=self.cfg.MODEL.STRIDE,
                                    mean=self.cfg.TEST.MEAN,
                                    std=self.cfg.TEST.STD)
                self.vis.images(vis_img, win=f"test_results[{i}]",
                                opts=dict(title=f"test_results[{i}]"))

            
            kps = get_kps_from_heatmap(results,
                                       self.cfg.MODEL.STRIDE,
                                       threshold=threshold,
                                       size=self.cfg.TEST.SIZE)
            dis, p, r = eval_key_points(kps, data['anns'], size=self.cfg.TEST.SIZE)
            
            eval_dis.update(dis.avg, dis.count)
            eval_p.update(p.avg, p.count)
            eval_r.update(r.avg, r.count)

        self.test_cnt += 1

        self.logger.info(f"Eval result: "
                         f"dis_loss: {eval_dis.avg:.2f} | "
                         f"precision: {eval_p.avg:.2%} | "
                         f"recall: {eval_r.avg:.2%}")

        # draw curve
        self.vis.line(Y=np.array([eval_dis.avg]),
                      X=np.array([self.test_cnt]),
                      win='test_dis_loss',
                      update=None if self.test_cnt == 1 else 'append',
                      opts=dict(title='test_dis_loss'))
        self.vis.line(Y=np.array([eval_p.avg]),
                      X=np.array([self.test_cnt]),
                      win='test_precision',
                      update=None if self.test_cnt == 1 else 'append',
                      opts=dict(title='test_precision'))
        self.vis.line(Y=np.array([eval_r.avg]),
                      X=np.array([self.test_cnt]),
                      win='test_recall',
                      update=None if self.test_cnt == 1 else 'append',
                      opts=dict(title='test_recall'))

        self.model.train()

