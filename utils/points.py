#!/usr/bin/env python
# coding=utf-8

import math

class Point:
    def __init__(self, x=0, y=0, score=1, radius=-1, angle=-1, cls=0, from_net=False):
        self.x = x
        self.y = y
        self.score = score
        self.radius = radius
        self.angle = angle
        self.cls = cls
        self.from_net = from_net

        if angle != -1:
            self.angle_rad = self.angle / 180.0 * math.pi
            self.sin = math.sin(self.angle_rad) 
            self.cos = math.cos(self.angle_rad)
        else:
            self.sin = 0
            self.cos = 1

    def __str__(self):
        string = ( f"\n\t\t( Point({self.x}, {self.y}): cls={self.cls}, score={self.score}, "
                   f"radius={self.radius}, angle={self.angle}, "
                   f"from_net={self.from_net} )" )
        return string
    
    __repr__ = __str__

