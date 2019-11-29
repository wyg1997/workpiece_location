#!/usr/bin/env python
# coding=utf-8

import torch
import numpy as np

class DataContainer:
    """
    A container include the data which are not wanted to be packed.
    """
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    @property
    def type(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        elif isinstance(self.data, np.ndarray):
            return self.data.type()
        else:
            return type(self.data)
