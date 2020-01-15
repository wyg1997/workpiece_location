#!/usr/bin/env python
# coding=utf-8

import xml.etree.ElementTree as ET

import numpy as np

from utils.templates import Template
from utils.cprint import cprint
from utils.points import Point


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


def calc_vector(kps):
    """
    Calculate vector between keypoints.
    """
    n_kps = len(kps)
    vec = np.ndarray((n_kps, n_kps, 2))
    for i in range(n_kps):
        for j in range(n_kps):
            if i == j:
                vec[i, i] = [kps[i].angle, kps[j].radius]
            else:
                vec[i, j] = [kps[j].x - kps[i].x, kps[j].y - kps[i].y]
    return vec


def check_self(p1, p2):
    """
    Check point info [angle, radius].
    """
    # angle
    if p1[0] != -1:
        diff_angle = abs(p1[0] - p2[0])
        diff_angle = min(diff_angle, 360-diff_angle)
        # TODO: angle threshold
        if diff_angle > 5:
            return False

    if p1[1] != -1:
        diff_radius = abs(p1[1] - p2[1])
        # TODO: radius threshold
        if diff_radius > 5:
            return False
    return True

def single_match(template, kps):
    des = template.compile()
    vec = calc_vector(kps)

    dis_thresh = template.offset ** 2
    n_trans = template.n_scale * template.n_rotate
    n_node = template.n_node
    n_points = len(kps)
    need_match = template.thresh * n_node
    if need_match == n_node:
        need_match -= 1

    group = []
    scores = []
    index = []

    # for each transform
    for i in range(n_trans):
        # As base point in n_node
        for node in range(n_node):
            # As base point in n_points
            for root in range(n_points):
                # different class, continue
                if template.nodes[node].cls != kps[root].cls:
                    continue
                # check point info with [angle, radius]
                if not check_self(des[i, node, node], vec[root, root]):
                    continue

                res = [-1] * n_node
                res[node] = root
                all_dis = 0
                find = 1

                # next class
                for next_node in range(n_node):
                    if node == next_node:
                        continue
                    best_ne = -1
                    min_dis = dis_thresh
                    # for other points and find best
                    for ne in range(n_points):
                        if ne in res:
                            continue
                        # different class
                        if template.nodes[next_node].cls != kps[ne].cls:
                            continue
                        # check point info with [angle, radius]
                        if not check_self(des[i, next_node, next_node], vec[ne, ne]):
                            continue
                        v1 = des[i, node, next_node]
                        v2 = vec[root, ne]
                        v = [v1[0] - v2[0], v1[1] - v2[1]]
                        dis = v[0] ** 2 + v[1] ** 2
                        if dis <= min_dis:
                            best_ne = ne
                            min_dis = dis
                    # save best point
                    res[next_node] = best_ne
                    find += 0 if best_ne == -1 else 1
                    all_dis += min_dis
                if find > need_match:
                    group.append(res)
                    scores.append(all_dis)
                    index.append(i)

    # choose from group
    order = np.argsort(scores)
    vis_node = []
    res = []
    for idx in order:
        if template.multi_match != -1 and len(res) >= template.multi_match:
            break
        find = 0
        for pos, node in enumerate(group[idx]):
            if node != -1 and node not in vis_node:
                find += 1
            else:
                group[idx][pos] = -1
        if find > need_match and group[idx] not in res:
            for node in group[idx]:
                if node != -1:
                    vis_node.append(node)
            res.append(idx)

    # output
    match_res = []

    # pick up model
    for idx in res:
        g = group[idx]
        tr = index[idx]
        s, a = template.get_affine(index[idx])
        p = template.get_trans_loc(s, a)

        temp_base = [0, 0]
        base = [0, 0]

        num_pick = 0
        for i_node, pos in enumerate(g):
            if pos != -1:
                temp_base[0] += p[i_node, 0]
                temp_base[1] += p[i_node, 1]
                base[0] += kps[pos].x
                base[1] += kps[pos].y
                num_pick += 1
        temp_base[0] /= num_pick
        temp_base[1] /= num_pick
        base[0] /= num_pick
        base[1] /= num_pick
        # find points which did not detect
        temp = []
        for i in range(n_node):
            x = int(base[0] + p[i, 0] - temp_base[0] + 0.5)
            y = int(base[1] + p[i, 1] - temp_base[1] + 0.5)
            cls = template.nodes[i].cls
            from_net = True if g[i] != -1 else False
            if not from_net:
                angle = template.nodes[i].angle
                angle = (angle - a) % 360 if angle != -1 else -1
                radius = template.nodes[i].radius
                radius = radius * s if radius != -1 else -1
                temp.append(Point(x=x, y=y, cls=cls, from_net=from_net,
                                  angle=angle, radius=radius))
            else:
                temp.append(Point(x=x, y=y, cls=cls, from_net=from_net))
        match_res.append(temp)

    return match_res


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    cat2label = {'lb': 0, 'lt': 1, 'rt': 2, 'rb': 3}
    templates = read_model_from_xml(
                    '/root/work/workpiece_location/example/type-c/v1/templates.xml',
                    cat2label
                )

    kps = []
    kps.append(Point(x=50, y=50, cls=1, radius=50, angle=0, from_net=True))
    kps.append(Point(x=150, y=50, cls=2, radius=50, angle=270, from_net=True))
    kps.append(Point(x=150, y=100, cls=3, radius=50, angle=180, from_net=True))
    kps.append(Point(x=50, y=100, cls=0, radius=50, angle=90, from_net=True))

    # cprint(templates[0], level='debug')
    # cprint(kps, level='debug')

    import time
    # compile template
    st = time.time()
    templates[0].compile()
    cprint(f"compile time: {time.time() - st}", level='debug')

    # single match
    st = time.time()
    cprint(single_match(templates[0], kps), level='debug')
    cprint(f"match time: {time.time() - st}", level='debug')

