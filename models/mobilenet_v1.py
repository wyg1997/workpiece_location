#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn

from utils.cprint import cprint


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )

class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, train_angle, train_size):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.angle_maps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, 2, kernel_size=1, padding=0, bn=False, relu=False)
        ) if train_angle else None

    def forward(self, x):
        trunk_features = self.trunk(x)

        outputs = {}
        # location
        outputs['locations'] = self.heatmaps(trunk_features)
        # angle
        if self.angle_maps is not None:
            outputs['angles'] = self.angle_maps(trunk_features)

        return outputs


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, train_angle, train_size):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.angle_maps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, 2, kernel_size=1, padding=0, bn=False, relu=False)
        ) if train_angle else None

    def forward(self, x):
        trunk_features = self.trunk(x)

        outputs = {}
        # location
        outputs['locations'] = self.heatmaps(trunk_features)
        # angle
        if self.angle_maps is not None:
            outputs['angles'] = self.angle_maps(trunk_features)
        return outputs


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, model_cfg, num_refinement_stages=1, num_channels=128, num_heatmaps=19):
        super().__init__()
        self.model_cfg = model_cfg

        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels,
                                          num_heatmaps,
                                          self.model_cfg.ANGLE,
                                          self.model_cfg.SIZE)
        self.refinement_stages = nn.ModuleList()

        # calculate channels
        refine_channels = num_channels + num_heatmaps
        if self.model_cfg.ANGLE:
            refine_channels += 2
        if self.model_cfg.SIZE:
            refine_channels += 1

        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(refine_channels,
                                                          num_channels,
                                                          num_heatmaps,
                                                          self.model_cfg.ANGLE,
                                                          self.model_cfg.SIZE))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        outputs = {'locations': []}
        if self.model_cfg.ANGLE:
            outputs['angles'] = []
        if self.model_cfg.SIZE:
            outputs['sizes'] = []

        # initial stage
        initial_stage_outputs = self.initial_stage(backbone_features)
        for key in outputs.keys():
            outputs[key].append(initial_stage_outputs[key])

        # refinement stage
        for refinement_stage in self.refinement_stages:
            latest_res = outputs['locations'][-1]
            if self.model_cfg.ANGLE:
                latest_res = torch.cat([latest_res, outputs['angles'][-1]], dim=1)
            if self.model_cfg.SIZE:
                latest_res = torch.cat([latest_res, outputs['sizes'][-1]], dim=1)

            refinement_stage_outputs = refinement_stage(torch.cat([backbone_features, latest_res],
                                                        dim=1))
            for key in outputs.keys():
                outputs[key].append(refinement_stage_outputs[key])

        return outputs

def build_mobilenet_v1(model_cfg, pretrain, nparts):
    model = PoseEstimationWithMobileNet(model_cfg,
                                        num_refinement_stages=2,
                                        num_channels=128,
                                        num_heatmaps=nparts)

    if pretrain:
        weights = torch.load('pretrain/mobilenet_v1.pth')
        model.load_state_dict(weights, strict=False)

    return model

