#!/usr/bin/env python
# coding=utf-8

from yacs.config import CfgNode as CN

#############
# deinition #
#############

_C = CN()


#########
# model #
#########

# base infomation
_C.MODEL = CN()
_C.MODEL.NAME = 'baseline'
_C.MODEL.VERSION = 'v0'
# backbone
# TODO: support other backbone
_C.MODEL.BACKBONE = 'resnet50'
# pretrain
# TODO: useless
_C.MODEL.PRETRAIN = True
# TODO: load checkpoint and continue train
_C.MODEL.RESUME = ''
_C.MODEL.CHECKPOINT = ''
# detect size and direction
# TODO: support detect size and direction
_C.MODEL.size = False
_C.MODEL.DIRECTION = False


################
# train images #
################

_C.TRAIN = CN()
# path
_C.TRAIN.DATA_ROOT = ''
# type in ['png', 'jpg', 'bmp']
_C.TRAIN.IMG_TYPE = 'png'
# image index(optional, set None will train all images)
_C.TRAIN.IMG_INDEX = []
# size: [h, w]
_C.TRAIN.SIZE = [512, 512]
# TODO: do flip
_C.TRAIN.DO_FLIP = False
_C.TRAIN.PROB = 0.5
# do image normalization
_C.TRAIN.MEAN = [0.485, 0.456, 0.406]
_C.TRAIN.STD = [0.229, 0.224, 0.225]
# random lightning and contrast
_C.TRAIN.DO_LIGHTNING = False
_C.TRAIN.MAX_LIGHTNING = 0.2
_C.TRAIN.P_LIGHTNING = 0.75
# heatmap parameters
_C.TRAIN.STRIDE = 1
_C.TRAIN.SIGMA = 5


###############
# test images #
###############

_C.TEST = CN()
# path
_C.TEST.DATA_ROOT = ''
# type in ['png', 'jpg', 'bmp']
_C.TEST.IMG_TYPE = 'png'
# image index(optional, set None will train all images)
_C.TEST.IMG_INDEX = []
# size: [h, w]
_C.TEST.SIZE = [512, 512]
# TODO: do flip
_C.TEST.DO_FLIP = False
_C.TEST.PROB = 0.5
# do image normalization
_C.TEST.MEAN = [0.485, 0.456, 0.406]
_C.TEST.STD = [0.229, 0.224, 0.225]
# random lightning and contrast
_C.TEST.DO_LIGHTNING = False
_C.TEST.MAX_LIGHTNING = 0.2
_C.TEST.P_LIGHTNING = 0.75


##############
# DataLoader #
##############

_C.DATALOADER = CN()
# TODO: more batch-size
_C.DATALOADER.BATCH_SIZE = 1
_C.DATALOADER.WORKERS = 8


##########
# solver #
##########

_C.SOLVER = CN()
# TODO: not supporting
_C.SOLVER.OPT = 'adam'
_C.SOLVER.EPOCHS = 30
# TODO: not supporting
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
# step
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [12, 24]
# warm up
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = 'linear'
# log
_C.SOLVER.LOG_INTERVAL = 10
_C.SOLVER.EVAL_EPOCH = 20


##########
# output #
##########

_C.OUTPUT_DIR = 'example/'


##########
# others #
##########

_C.VISDOM = CN()
_C.VISDOM.PORT = 8097