# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset.py
# Description :preprocess data
# --------------------------------------

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import os
from PIL import Image
import math
import numpy as np
import tensorflow as tf
from cfg.config import path_params, model_params, classes_map
from utils.process_utils import *

class Dataset(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.anchors = model_params['anchors']
        self.image_height = model_params['image_height']
        self.image_width = model_params['image_width']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.iou_threshold = model_params['iou_threshold']

    def letterbox_resize(self, image, bboxes, new_height, new_width, interp=0):
        """
        Resize the image and correct the bbox accordingly.
        :param image: BGR image data shape is [height, width, channel]
        :param bboxes: bounding box shape is [num, 4]
        :param new_height: new image height
        :param new_width: new image width
        :param interp:
        :return: result
        """
        origin_height, origin_width = image.shape[:2]
        resize_ratio = min(new_width / origin_width, new_height / origin_height)
        resize_width = int(resize_ratio * origin_width)
        resize_height = int(resize_ratio * origin_height)

        image = cv2.resize(image, (resize_width, resize_height), interpolation=interp)
        image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

        dw = int((new_width - resize_width) / 2)
        dh = int((new_height - resize_height) / 2)

        image_padded[dh:resize_height + dh, dw:resize_width + dw, :] = image

        # xmin, xmax, ymin, ymax
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh

        return image_padded, bboxes

    def load_data(self, filename):
        image_path = os.path.join(self.data_path, "images", filename+'.jpg')
        image = cv2.imread(image_path)

        label_path = os.path.join(self.data_path, "labels", filename+'.txt')
        lines = [line.rstrip() for line in open(label_path)]
        num_obj = len(lines)

        index = 0
        bboxes = np.zeros([num_obj, (4 + 1)], dtype=np.float32)
        for line in lines:
            data = line.split(' ')
            data[0:] = [float(t) for t in data[0:]]
            bboxes[index, 0] = data[0]
            bboxes[index, 1] = data[1] - data[3] / 2.0
            bboxes[index, 2] = data[2] - data[4] / 2.0
            bboxes[index, 3] = data[1] + data[3] / 2.0
            bboxes[index, 4] = data[2] + data[4] / 2.0
            index += 1

        image, bboxes = self.letterbox_resize(image, bboxes, self.image_height, self. image_width)

        return image, bboxes



