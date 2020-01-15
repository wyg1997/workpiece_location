#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np

from .points import Point


class Template:
    def __init__(self, multi_match=1, offset=0.5, thresh=10, scale=[20, 100], rotate=[-90, 90], nodes=[]):
        self.multi_match = multi_match
        self.offset = offset
        self.thresh = thresh
        self.scale = scale
        self.rotate = rotate
        self.nodes = nodes

        self.loc = self._get_loc_matrix(self.nodes)

        self.n_scale = self.scale[1] - self.scale[0] + 1
        self.n_rotate = self.rotate[1] - self.rotate[0] + 1

        self.compiled = False

    def _get_loc_matrix(self, nodes):
        """
        Put nodes into matrix with shape [n, 3].
        """
        loc = np.ndarray((len(nodes), 3))
        for idx, node in enumerate(nodes):
            loc[idx] = [node.x, node.y, 1]
        return loc

    def get_trans_loc(self, s, a):
        # rotate and scale
        M = cv2.getRotationMatrix2D((0, 0), a, scale=s)
        p = np.matmul(M, self.loc.T).T
        return p


    def get_affine(self, n):
        """
        Get affine from index ( scale and rotate ).
        `n` start with 0.
        """
        s = n // self.n_rotate + self.scale[0]
        a = n % self.n_rotate + self.rotate[0]
        return s, a

    def compile(self):
        """
        Calculate desription of template with shape [n, node_from, node_to, 2].
        If node_from == node_to, it means [angle, radius] ( -1 mean all ).
        """
        if not self.compiled:
            self.des = np.ndarray((self.n_scale*self.n_rotate, self.n_node, self.n_node, 2))
            for i_s in range(self.n_scale):
                for i_rot in range(self.n_rotate):
                    s = self.scale[0] + i_s
                    rot = self.rotate[0] + i_rot
                    p = self.get_trans_loc(s, rot)  # get transformed locations with shape [n, 2] 
                    idx = i_s * self.n_rotate + i_rot  # get index
                    for i in range(self.n_node):
                        for j in range(self.n_node):
                            if i == j:
                                # self angle
                                if self.nodes[i].angle == -1:
                                    self.des[idx, i, i, 0] = -1
                                else:
                                    self.des[idx, i, i, 0] = (self.nodes[i].angle - rot) % 360
                                # self radius
                                if self.nodes[i].radius == -1:
                                    self.des[idx, i, i, 1] = -1
                                else:
                                    self.des[idx, i, i, 1] = self.nodes[i].radius * s
                            else:
                                # vector node[i] -> node[j]
                                self.des[idx, i, j] = [p[j, 0] - p[i, 0], p[j, 1] - p[i, 1]]
            self.compiled = True
        return self.des

    @property
    def n_node(self):
        return len(self.nodes)

    def __str__(self):
        string = f"""\nTemplate:
\tglobal setting:
\t\tmulti-match={self.multi_match}
\t\tthresh={self.thresh}
\t\tscale={self.scale}
\t\trotate={self.rotate}
\tNodes:
\t\t{self.nodes}
"""
        return string

    __repr__ = __str__

