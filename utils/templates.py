#!/usr/bin/env python
# coding=utf-8

import math
from copy import deepcopy

import cv2
import numpy as np
import xml.etree.ElementTree as ET

from utils.points import Point
from utils.cprint import cprint


class TemplateMatchTool:
    '''
    Affine matrix:
        [ a c e
          b d f
          0 0 1 ]
    and(only has shear_x):
        a = scale_x * cos(angle)
        b = scale_x * sin(angle)
        c = shear_x * scale_x * cos(angle) - scale_y * sin(angle)
        d = shear_y * scale_x * sin(angle) + scale_y * cos(angle)
        e = shift_x
        f = shift_y
    so we can get:
        angle = atan2(b, a)
        denom(assist) = a*a + b*b
        scale_x = sqrt(denom)
        scale_y = (a*d - c*b) / scale_x
        shear_x = atan2(a*c + b*d, denom)
        shear_y(useless) = 0
        shift_x = e
        shift_y = f

    Affine transform order is:
        origin -> shearX -> scale(x, y) -> rotate -> shift -> target
    matrix of each affine transformation:
        shearX:
            [ 1 tan(a) 0
              0    1   0
              0    0   1 ]
        scale(x, y):
            [ scale_x    0    0
                0     scale_y 0
                0        0    1 ]
        rotate(counter clockwise):
            [ cos(angle) -sin(angle) 0 
              sin(angle)  cos(angle) 0
                   0          0      1 ]
        shift(x, y):
            [ 1 0 shift_x
              0 1 shift_y
              0 0    1    ]
    '''
    class Affine:
        angle = 0  # rad
        scale_x = 1
        scale_y = 1
        shear_x = 0
        shear_y = 0  # always zero
        shift_x = 0
        shift_y = 0

        @property
        def degree(self):
            return self.angle*180/math.pi

        def __str__(self):
            string = f"angle: {self.degree:.3f} | scale_x: {self.scale_x:.3f} | " \
                     f"scale_y: {self.scale_y:.3f} | shear_x: {self.shear_x:.3f} | " \
                     f"shift_x: {self.shift_x:.3f} | shift_y: {self.shift_y:.3f}"
            return string

        __repr__ = __str__


    
    __ratio_thresh = 1.2
    __shear_thresh = 0.1  # about 5 degree

    # 类别过滤结果
    __point_set = []
    # 候选点集
    __candidate = []

    def __init(self):
        self.__point_set = []
        self.__candidate = []

    def __dfs(self, now_set, vis, got, pos, n_node, point_thresh):
        if pos == n_node:
            if got > point_thresh:
                self.__candidate.append(now_set.copy())
            return None
        # 后面的点都加入点集也不够，直接退出
        if got+n_node-pos < point_thresh:
            return None

        for idx in self.__point_set[pos]:
            if not vis[idx]:
                vis[idx] = True
                now_set[pos] = idx
                self.__dfs(now_set, vis, got+1, pos+1, n_node, point_thresh)
                vis[idx] = False
        now_set[pos] = -1
        self.__dfs(now_set, vis, got, pos+1, n_node, point_thresh)

    def __point_list_to_matrix(self, point_list):
        '''
        Transform List[Point, Point, ...] to numpy.array with shape [n, 2]
        '''
        n = len(point_list)
        res = np.ndarray((n, 2), dtype=np.float)
        for i in range(n):
            res[i, 0] = point_list[i].x
            res[i, 1] = point_list[i].y
        return res

    
    def __fit_affine_matrix(self, origin, result):
        origin = np.insert(origin, 2, 1, axis=1)  # to [n, 3]
        affine_matrix = np.linalg.pinv(origin.T @ origin) @ origin.T @ result
        return affine_matrix  # [3, 2]

    def __get_affine_param(self, affine_matrix):
        a = affine_matrix[0, 0]
        b = affine_matrix[0, 1]
        c = affine_matrix[1, 0]
        d = affine_matrix[1, 1]
        e = affine_matrix[2, 0]
        f = affine_matrix[2, 1]
        affine = self.Affine()
        affine.angle = math.atan2(b, a)
        denom = a*a + b*b
        affine.scale_x = math.sqrt(denom)
        affine.scale_y = (a*d - c*b) / (affine.scale_x + 1e-8)
        affine.shear_x = math.atan2(a*c + b*d, denom)
        affine.shift_x = e
        affine.shift_y = f
        return affine

    def __get_affine_matrix(self, affine):
        matrix = np.ndarray((3, 2), np.float)
        matrix[0, 0] = affine.scale_x * math.cos(affine.angle)
        matrix[0, 1] = affine.scale_x * math.sin(affine.angle)
        matrix[1, 0] = affine.shear_x * affine.scale_x * math.cos(affine.angle) - \
                       affine.scale_y * math.sin(affine.angle)
        matrix[1, 1] = affine.scale_y * math.cos(affine.angle)
        matrix[2, 0] = affine.shift_x
        matrix[2, 1] = affine.shift_y
        return matrix

    def single_match(self, template, kps):
        self.__init()
        n_node = template.n_node
        n_kps = len(kps)
        point_thresh = n_node * template.thresh

        # step 1: get canditate
        for i in range(n_node):
            s = []
            for j in range(len(kps)):
                if kps[j].cls == template.nodes[i].cls:
                    s.append(j)
            self.__point_set.append(s)
        now_set = [-1] * n_node
        vis = [False] * n_kps
        self.__dfs(now_set, vis, 0, 0, n_node, point_thresh)
        if len(self.__candidate) == 0:
            return []

        # step 2: calculate score of each candidate
        temp_candidate = []
        for l in self.__candidate:
            point_list = []
            model_list = []
            for i_tmpl, i_kps in enumerate(l):
                if i_kps != -1:
                    model_list.append(template.nodes[i_tmpl])
                    point_list.append(kps[i_kps])
            pm = self.__point_list_to_matrix(point_list)  # point matrix
            mm = self.__point_list_to_matrix(model_list)  # model matrix
            affine_matrix = self.__fit_affine_matrix(mm, pm)
            affine_param = self.__get_affine_param(affine_matrix)

            # check ratio and shear_x
            if affine_param.scale_x < 0 or affine_param.scale_y < 0:
                continue
            # the min scale is 1
            if affine_param.scale_x < 1 or affine_param.scale_y < 1:
                affine_param.scale_x = max(affine_param.scale_x, 1)
                affine_param.scale_y = max(affine_param.scale_y, 1)
            else:
                ratio = affine_param.scale_x / affine_param.scale_y
                if max(ratio, 1/ratio) > self.__ratio_thresh or \
                   abs(affine_param.shear_x) > self.__shear_thresh:
                    continue
            # check each points' infos
            score = 1.0
            pass_check = True
            for i_tmpl, i_kps in enumerate(l):
                if i_kps == -1:
                    score -= 1.0 / n_node
                else:
                    # angle
                    if template.nodes[i_tmpl].angle != -1 and \
                       kps[i_kps].angle != -1:
                        offset = (affine_param.degree + template.nodes[i_tmpl].angle - \
                                  kps[i_kps].angle) % 360
                        offset = min(offset, 360-offset)
                        if offset > 5:
                            pass_check = False
                            break
                        score -= offset / 5 / n_node
                    # radius
                    if template.nodes[i_tmpl].radius != -1 and \
                       kps[i_kps].radius != -1:
                        offset = abs(afffine_param.scale_x * template.nodes[i_tmpl].radius - \
                                  kps[i_kps].radius)
                        if offset > 10:
                            pass_check = False
                            break
                        score -= offset / 10 / n_node
            if not pass_check:
                break
            # check locations
            loc_offset = np.insert(mm, 2, 1, axis=1) @ affine_matrix - pm
            loc_offset = (loc_offset * loc_offset).sum(axis=1)
            if loc_offset.max() > template.offset:
                break
            score -= loc_offset.sum() / template.offset / n_node
            # pass check
            if score >= template.thresh:
                temp_candidate.append((l, affine_matrix, affine_param, score))
        self.__candidate = temp_candidate

        # step 3: sort and get result
        result = []
        self.__candidate = sorted(self.__candidate, key=lambda x: x[3], reverse=True)
        vis = [False] * n_kps
        for ids, affine_matrix, afffine_param, score in self.__candidate:
            res = []
            # check this point whether used or not
            check = True
            for idx in ids:
                if idx != -1 and vis[idx]:
                    check = False
                    break
            if not check:
                continue
            # get result
            loc = np.insert(template.matrix, 2, 1, axis=1) @ affine_matrix  # [n, 2]
            for i, idx in enumerate(ids):
                if idx == -1:
                    p = Point(x=loc[i, 0], y=loc[i, 1], cls=template.nodes[i].cls,
                              from_net=False)
                    if template.nodes[i].angle != -1:
                        p.angle = (template.nodes[i].angle + affine_param.degree) % 360
                    if template.nodes[i].radius != -1:
                        p.radius = template.nodes[i].radius * affine_param.scale_x
                else:
                    p = deepcopy(kps[idx])
                    p.shift_x = loc[i, 0] - p.x
                    p.shift_y = loc[i, 1] - p.y
                res.append(p)
            result.append(res)
            # set vis flag
            for idx in ids:
                if idx != -1:
                    vis[idx] = True
        return result


class Template:
    def __init__(self, multi_match=1, offset=0.5, thresh=10, scale=[20, 100], rotate=[-90, 90], nodes=[]):
        self.multi_match = multi_match
        self.offset = offset
        self.thresh = thresh
        self.scale = scale
        self.rotate = rotate
        self.nodes = nodes

        # get node matrix with shape [n, 2]
        n_node = len(self.nodes)
        self.matrix = np.ndarray((n_node, 2), np.float)
        for i in range(n_node):
            self.matrix[i, 0] = self.nodes[i].x
            self.matrix[i, 1] = self.nodes[i].y


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



def read_model_from_xml(model_path, cat2label):
    """
    Read templates from xml file.
    """
    templates = []

    root = ET.parse(model_path).getroot()
    for obj in root.findall('object'):
        settings = obj.find('global')

        multi_match = int(settings.find('multi-match').text)
        offset = float(settings.find('offset').text)
        thresh = float(settings.find('thresh').text)
        # scale
        s = settings.find('scale')
        scale = [
            int(s.find('scale-min').text),
            int(s.find('scale-max').text)
        ]
        # rotate
        r = settings.find('rotate')
        rotate = [
            int(r.find('rotate-min').text),
            int(r.find('rotate-max').text)
        ]

        # nodes
        nodes = []
        ns = obj.find('nodes')
        for n in ns.findall('node'):
            x = float(n.find('x').text)
            y = float(n.find('y').text)
            cls = cat2label[n.find('cls').text]
            radius = float(n.find('radius').text)
            angle = float(n.find('angle').text)

            node = Point(x=x, y=y, cls=cls, radius=radius, angle=angle)
            nodes.append(node)
        temp = Template(multi_match, offset, thresh, scale, rotate, nodes)
        templates.append(temp)

    return templates


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    cat2label = {'0': 0, '1': 1, '2': 2}
    templates = read_model_from_xml(
                    '/root/work/workpiece_location/example/type-c/v1/templates_temp.xml',
                    cat2label
                )

    kps = []
    # kps.append(Point(x=70, y=70, cls=0, from_net=True))
    # kps.append(Point(x=170, y=70, cls=1, from_net=True))
    # kps.append(Point(x=170, y=120, cls=2, from_net=True))

    # kps.append(Point(x=270, y=270, cls=0, from_net=True))
    # kps.append(Point(x=370, y=270, cls=1, from_net=True))
    # kps.append(Point(x=370, y=320, cls=2, from_net=True))

    # kps.append(Point(x=100, y=100, cls=1, from_net=True))
    # kps.append(Point(x=100, y=200, cls=0, from_net=True))
    # kps.append(Point(x=150, y=100, cls=2, from_net=True))

    # kps.append(Point(x=150, y=300, cls=0, from_net=True))
    # kps.append(Point(x=240, y=255, cls=1, from_net=True))
    # kps.append(Point(x=262, y=300, cls=2, from_net=True))

    # kps.append(Point(x=150, y=50, cls=0, from_net=True))
    # kps.append(Point(x=350, y=50, cls=1, from_net=True))
    # kps.append(Point(x=350, y=150, cls=2, from_net=True))

    # kps.append(Point(x=300, y=300, cls=2, from_net=True))
    # kps.append(Point(x=350, y=300, cls=1, from_net=True))
    # kps.append(Point(x=350, y=200, cls=0, from_net=True))

    # kps.append(Point(x=500, y=100, cls=0, from_net=True))
    # kps.append(Point(x=500, y=200, cls=1, from_net=True))

    kps.append(Point(x=100, y=200, cls=2, from_net=True))
    kps.append(Point(x=39, y=139, cls=1, from_net=True))

    # cprint(templates[0], level='debug')
    # cprint(kps, level='debug')

    import time

    mt = TemplateMatchTool()

    # single match
    st = time.time()
    res = mt.single_match(templates[0], kps)
    cprint(f"match time: {time.time() - st}", level='debug')
    cprint(res, level='debug')

    import cv2
    def draw_point(img, point):
        color = (0, 1, 127/255)[::-1]
        img = cv2.circle(img, (point.x, point.y), 5, color, 1, 1)
        return img

    h, w = 400, 600
    img = np.ndarray((h, w, 3))
    # draw points
    for p in kps:
        img = draw_point(img, p)
    # draw results
    for group in res:
        l = len(group)
        lines = []
        for p in group:
            lines.append([int(p.x), int(p.y)])
        img = cv2.polylines(img,
                            [np.array(lines)],
                            isClosed=True,
                            color=(0.0, 191/255, 1.0)[::-1],
                            thickness=2)

    img = (img*255).astype(np.int)
    cv2.imwrite('/root/work/temp/match_demo.png', img)
    cprint("imwrite ok!", level='warn')

