#!/usr/bin/env python
# coding=utf-8

import sys
sys.path.append('.')
import os.path as osp
import argparse

import cv2
import xml.etree.ElementTree as ET

from utils.cprint import cprint


# make xml format more beautiful
def indent(elem, level=0):
    i = '\n' + level*'\t'
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + '\t'
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write2xml(labels, root_path, idx, img_type):
    img = cv2.imread(osp.join(root_path, 'source', f"{idx}.{img_type}"))
    h, w = img.shape[-2:]

    # write to xml
    root = ET.Element('annotation')

    # size
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)

    for label in labels:
        cls, x, y = label
        print(label)

        obj = ET.SubElement(root, 'point')

        # label
        label = ET.SubElement(obj, 'name')
        label.text = cls

        loc = ET.SubElement(obj, 'location')

        # cor
        xx = ET.SubElement(loc, 'x')
        yy = ET.SubElement(loc, 'y')
        xx.text = str(x)
        yy.text = str(y)

        # size
        s = ET.SubElement(obj, 'size')
        s.text = '40'
        # angle
        d = ET.SubElement(obj, 'angle')
        d.text = '0'

    # make file more beautiful
    indent(root)
    cprint(ET.tostring(root, 'utf-8').decode(), level='debug')

    # save
    tree = ET.ElementTree(root)
    tree.write(osp.join(root_path, 'label', f"{idx}.xml"), encoding='utf-8', xml_declaration=True)


def parse_args():
    parser = argparse.ArgumentParser(description='key points marking tool')
    parser.add_argument(
        '--classes',
        help='All class names',
        type=str,
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--path',
        help='Data root path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--idx',
        help='The image id at first',
        type=int,
        default=1
    )
    parser.add_argument(
        '--img_type',
        choices=['png', 'jpg', 'bmp'],
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    classes = args.classes
    id2cls = {}
    for i, cls in enumerate(classes):
        id2cls[i+1] = cls

    path = args.path
    idx = args.idx
    img_type = args.img_type

    while(True):
        cprint(f"=== Now image id is {idx} ===", level='warn')
        
        cprint(f"{id2cls}", level='debug')

        labels = []
        ans = input(f"Please type in class_index and x y(eg: 1 60 168) or finish(enter) or exit(q): ")
        if ans == 'q':
            break
        while ans != '':
            cls_id, x, y = ans.strip().split(' ')
            cls = id2cls[int(cls_id)]
            x = int(x)
            y = int(y)

            labels.append([cls, x, y])

            ans = input(f"Please type in class_index and x y(eg: 1 60 168) or finish(enter): ")

        write2xml(labels, path, idx, img_type)

        idx += 1
        print()

    cprint(f"finish labeling...", level='debug')
        

if __name__ == '__main__':
    main()
