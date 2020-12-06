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
        self.num_classes = model_params['classes']
        self.input_height = model_params['input_height']
        self.input_width = model_params['input_width']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.iou_threshold = model_params['iou_threshold']

    def convert(self, data):
        x1 = data[1] - data[3] / 2.0
        y1 = data[2] - data[4] / 2.0
        x2 = data[1] + data[3] / 2.0
        y2 = data[2] + data[4] / 2.0
        class_id = data[0]

        return [x1, y1, x2, y2, class_id]

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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(self.data_path, "labels", filename+'.txt')
        lines = [line.rstrip() for line in open(label_path)]

        bboxes = []

        for line in lines:
            data = line.split(' ')
            data[0:] = [float(t) for t in data[0:]]
            box = self.convert(data)
            bboxes.append(box)

        image_rgb, bboxes = self.letterbox_resize(image_rgb, np.array(bboxes, dtype=np.float32), self.input_height, self. input_width)

        while bboxes.shape[0] < 150:
            bboxes = np.append(bboxes, [[0.0, 0.0, 0.0, 0.0, 0.0]], axis=0)

        bboxes = np.array(bboxes, dtype=np.float32)
        image_raw = image_rgb.tobytes()
        bbox_raw = bboxes.tobytes()

        return image_raw, bbox_raw

    def preprocess_true_boxes(self, labels):
        """
        preprocess true boxes to train input format
        :param labels: numpy.ndarray of shape [20, 5]
                       shape[0]: the number of labels in each image.
                       shape[1]: x_min, y_min, x_max, y_max, class_index, yaw
        :return: y_true shape is [feature_height, feature_width, per_anchor_num, 5 + num_classes]
        """
        # class id must be less than num_classes
        #assert (labels[..., 4] < len(self.num_classes)).all()
        input_shape = np.array([self.input_height, self.input_width], dtype=np.int32)

        assert input_shape[0] % 32 == 0
        assert input_shape[1] % 32 == 0

        feature_sizes = input_shape // 32

        # anchors 归一化到图像空间0~1
        num_anchors = len(self.anchors)
        anchor_array = np.array(model_params['anchors'])# / input_shape

        # labels 去除空标签
        valid = (np.sum(labels, axis=-1) > 0).tolist()
        labels = labels[valid]

        y_true = np.zeros(shape=[feature_sizes[0], feature_sizes[1], num_anchors, 4 + 1 + len(self.num_classes)], dtype=np.float32)

        boxes_xy = (labels[:, 0:2] + labels[:, 2:4]) / 2
        boxes_wh = labels[:, 2:4] - labels[:, 0:2]
        true_boxes = np.concatenate([boxes_xy, boxes_wh], axis=-1)

        anchors_max = anchor_array / 2.
        anchors_min = - anchor_array / 2.

        valid_mask = boxes_wh[:, 0] > 0
        wh = boxes_wh[valid_mask]

        # [N, 1, 2]
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = - wh / 2.

        # [N, 1, 2] & [5, 2] ==> [N, 5, 2]
        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        # [N, 5, 2]
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchor_array[:, 0] * anchor_array[:, 1]
        # [N, 5]
        iou = intersect_area / (box_area + anchor_area - intersect_area + tf.keras.backend.epsilon())

        # Find best anchor for each true box [N]
        best_anchor = np.argmax(iou, axis=-1)

        for t, k in enumerate(best_anchor):
            i = int(np.floor(true_boxes[t, 0] / 32.))
            j = int(np.floor(true_boxes[t, 1] / 32.))
            c = labels[t, 4].astype('int32')
            y_true[j, i, k, 0:4] = true_boxes[t, 0:4]
            y_true[j, i, k, 4] = 1
            y_true[j, i, k, 5 + c] = 1

        return y_true

