#!/usr/bin/env python
# coding=utf-8

from utils.cprint import cprint
from .resnet34 import build_resnet34

_ALL_MODELS = ['resnet34']

def build_model(model_name, num_cls, pretrain=False):
    if model_name not in _ALL_MODELS:
        cprint(f"model name must be one of {_ALL_MODELS}, ",
               f"but get {model_name}.",
               level='error')
        raise ValueError(f"model name {model_name} not support")

    if model_name == 'resnet34':
        model = build_resnet34(pretrain, nparts=num_cls)

    # check
    if model is None:
        cprint(f"model build failed...", level='error')
        raise ValueError(f"model `{model_name}` build error, please check again")
    return model
