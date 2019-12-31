#!/usr/bin/env python
# coding=utf-8

import os
import os.path as osp

import torch
import numpy as np
from tqdm import tqdm
import cv2
from prefetch_generator import BackgroundGenerator

from datasets.build import get_dataloader
from datasets.transforms import resume_imgs
from tools.kps_tools import get_kps_from_heatmap, eval_key_points, resize_heatmaps
from utils.meters import AverageMeter
from utils.cprint import cprint
from tools.visualize import vis_heatmaps, vis_results


class Tester:
    def __init__(self, cfg, logger, vis, work_dir, model, classes):
        self.cfg, self.logger, self.vis, self.work_dir = cfg, logger, vis, work_dir
        self.model, self.classes = model, classes
        self.num_cls = len(classes) + 1

        # task
        self.task = ['locations']
        if self.cfg.MODEL.ANGLE:
            self.task.append('angles')
        if self.cfg.MODEL.SIZE:
            self.task.append('sizes')

        self.dataset, self.test_dataloader = get_dataloader(cfg, kind='test', CLASSES=classes)

        self.test_cnt = 0
        self.save_cnt = 0
        self.save_dir = osp.join(self.work_dir, 'results/')
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def test(self, threshold=0.5, show=False):
        self.test_cnt += 1

        print()
        self.model.eval()

        eval_dis = AverageMeter()
        eval_p = AverageMeter()
        eval_r = AverageMeter()
        eval_angle_error = AverageMeter()
        eval_size_error = AverageMeter()

        for i, data in enumerate(BackgroundGenerator(tqdm(self.test_dataloader,
                                                          desc=f"Testing"))):
            ori_imgs = data['imgs']

            imgs = ori_imgs.cuda()
            with torch.no_grad():
                outputs = self.model(imgs)

            # collate outputs
            results = {}
            for key in self.task:
                results[key] = resize_heatmaps(outputs[key][-1], self.cfg.MODEL.STRIDE)
            
            kps = get_kps_from_heatmap(results,
                                       threshold=threshold,
                                       size=self.cfg.TEST.SIZE)
            # show results
            if show:
                ori_imgs = resume_imgs(ori_imgs, self.cfg.TEST.MEAN, self.cfg.TEST.STD)
                res_img = vis_results(np.copy(ori_imgs), kps, self.classes, self.cfg.VISDOM.SHOW_INFO)

                # # save results
                # save_img = res_img.transpose(0, 2, 3, 1)
                # for i_img in range(res_img.shape[0]):
                #     self.save_cnt += 1
                #     cv2.imwrite(osp.join(self.save_dir, f"{self.save_cnt}.png"), save_img[i_img, :, :, ::-1])

                # show results
                self.vis.images(res_img, win=f"test_results[{i}]",
                                opts=dict(title=f"test_results[{i}]"))

                # # heatmap
                # heat_img = vis_heatmaps(np.copy(ori_imgs), results['locations'], alpha=0.5)
                # self.vis.images(heat_img, win=f"test_result_heatmap[{i}]",
                #                 opts=dict(title=f"test_result_heatmap[{i}]"))

            eval_res = eval_key_points(kps, data['anns'], size=self.cfg.TEST.SIZE)
            
            eval_dis.update(eval_res['dis'].avg, eval_res['dis'].count)
            eval_p.update(eval_res['precision'].avg, eval_res['precision'].count)
            eval_r.update(eval_res['recall'].avg, eval_res['recall'].count)
            if 'angles' in self.task:
                eval_angle_error.update(eval_res['angle_error'].avg, eval_res['angle_error'].count)
            if 'sizes' in self.task:
                eval_size_error.update(eval_res['size_error'].avg, eval_res['size_error'].count)

        self.logger.info(
            f"Eval result: "
            f"dis_error: {eval_dis.avg:.2f} | "
            f"angle_error: {-1 if 'angles' not in self.task else eval_angle_error.avg:.4f} | "
            f"size_error: {-1 if 'sizes' not in self.task else eval_size_error.avg:.2f} | "
            f"precision: {eval_p.sum:.0f}/{eval_p.count}={eval_p.avg:.2%} | "
            f"recall: {eval_r.sum:.0f}/{eval_r.count}={eval_r.avg:.2%}")

        # draw curve
        self.vis.line(Y=np.array([eval_dis.avg]),
                      X=np.array([self.test_cnt]),
                      win='test_dis_error',
                      update=None if self.test_cnt == 1 else 'append',
                      opts=dict(title='test_dis_error'))
        self.vis.line(Y=np.array([eval_angle_error.avg]),
                      X=np.array([self.test_cnt]),
                      win='test_angle_error',
                      update=None if self.test_cnt == 1 else 'append',
                      opts=dict(title='test_angle_error'))
        self.vis.line(Y=np.array([eval_size_error.avg]),
                      X=np.array([self.test_cnt]),
                      win='test_size_error',
                      update=None if self.test_cnt == 1 else 'append',
                      opts=dict(title='test_size_error'))
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

