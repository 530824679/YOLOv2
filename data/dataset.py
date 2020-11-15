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
import math
import numpy as np
import tensorflow as tf
from cfg.config import data_params, path_params, model_params, classes_map, anchors
from utils.process_utils import *

class Dataset(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.anchors = anchors
        self.image_height = model_params['image_height']
        self.image_width = model_params['image_width']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.iou_threshold = model_params['iou_threshold']
        self.x_min = data_params['x_min']
        self.x_max = data_params['x_max']
        self.y_min = data_params['y_min']
        self.y_max = data_params['y_max']
        self.z_min = data_params['z_min']
        self.z_max = data_params['z_max']

    def load_bev_image(self, data_num):
        pcd_path = os.path.join(self.data_path, "object/training/livox", data_num+'.pcd')
        pts = self.load_pcd(pcd_path)
        roi_pts = self.filter_roi(pts)
        bev_image = self.transform_bev_image(roi_pts)
        return bev_image

    def load_bev_label(self, data_num):
        txt_path = os.path.join(self.data_path, "object/training/label", data_num + '.txt')
        label = self.load_label(txt_path)
        bev_label = self.transform_bev_label(label)
        encoded_label = self.encode(bev_label)
        return encoded_label

    def load_pcd(self, pcd_path):
        pts = []
        f = open(pcd_path, 'r')
        data = f.readlines()
        f.close()

        line = data[9].strip('\n')
        pts_num = eval(line.split(' ')[-1])

        for line in data[11:]:
            line = line.strip('\n')
            xyzi = line.split(' ')
            x, y, z, i = [eval(i) for i in xyzi[:4]]
            pts.append([x, y, z, i])

        assert len(pts) == pts_num
        res = np.zeros((pts_num, len(pts[0])), dtype=np.float)
        for i in range(pts_num):
            res[i] = pts[i]

        return res

    def calc_xyz(self, data):
        center_x = (data[16] + data[19] + data[22] + data[25]) / 4.0
        center_y = (data[17] + data[20] + data[23] + data[26]) / 4.0
        center_z = (data[18] + data[21] + data[24] + data[27]) / 4.0
        return center_x, center_y, center_z

    def calc_hwl(self, data):
        height = (data[15] - data[27])
        width = math.sqrt(math.pow((data[17] - data[26]), 2) + math.pow((data[16] - data[25]), 2))
        length = math.sqrt(math.pow((data[17] - data[20]), 2) + math.pow((data[16] - data[19]), 2))
        return height, width, length

    def calc_yaw(self, data):
        angle = math.atan2(data[17] - data[26], data[16] - data[25])

        if (angle < -1.57):
            return angle + 3.14 * 1.5
        else:
            return angle - 1.57

    def cls_type_to_id(self, data):
        type = data[1]
        if type not in classes_map.keys():
            return -1
        return classes_map[type]

    def load_label(self, label_path):
        lines = [line.rstrip() for line in open(label_path)]
        num_obj = len(lines)

        index = 0
        true_boxes = np.zeros([num_obj, (6 + 1 + 1)], dtype=np.float32)
        for line in lines:
            data = line.split(' ')
            data[4:] = [float(t) for t in data[4:]]
            true_boxes[index, 0] = self.cls_type_to_id(data)
            true_boxes[index, 1], true_boxes[index, 2], true_boxes[index, 3] = self.calc_xyz(data)
            true_boxes[index, 4], true_boxes[index, 5], true_boxes[index, 6] = self.calc_hwl(data)
            true_boxes[index, 7] = self.calc_yaw(data)
            index += 1

        return true_boxes

    def transform_bev_label(self, true_box):
        bev_height = model_params['image_height']
        bev_width = model_params['image_width']

        range_x = data_params['x_max'] - data_params['x_min']
        range_y = data_params['y_max'] - data_params['y_min']

        boxes_num = true_box.shape[0]

        num = 0
        # for obj in objects:
        for i in range(boxes_num):
            if (true_box[i][1] > data_params['x_min']) & (true_box[i][1] < data_params['x_max']) & \
                    (true_box[i][2] > data_params['y_min']) & (true_box[i][2] < data_params['y_max']):
                num = num + 1

        # true_box: class, x, y, z, h, w, l, rz
        # bev_box: class, x, y, w, l, rz
        bev_box = np.zeros([num, 6], dtype=np.float32)

        index = 0
        for j in range(boxes_num):
            if (true_box[j][1] > data_params['x_min']) & (true_box[j][1] < data_params['x_max']) & (
                    true_box[j][2] > data_params['y_min']) & (true_box[j][2] < data_params['y_max']):
                bev_box[index][0] = true_box[j][0]
                bev_box[index][1] = (true_box[j][2] + 0.5 * range_y) / range_y * bev_width
                bev_box[index][2] = true_box[j][1] / range_x * bev_height
                bev_box[index][3] = true_box[j][5] / range_y * bev_width
                bev_box[index][4] = true_box[j][6] / range_x * bev_height
                bev_box[index][5] = true_box[j][7]
                index = index + 1

        return bev_box

    def transform_bev_image(self, pts):
        bev_height = model_params['image_height']
        bev_width = model_params['image_width']

        range_x = data_params['x_max'] - data_params['x_min']
        range_y = data_params['y_max'] - data_params['y_min']

        # Discretize Feature Map
        point_cloud = np.copy(pts)
        point_cloud[:, 0] = np.int_(np.floor(point_cloud[:, 0] / range_x * (bev_height - 1)))
        point_cloud[:, 1] = np.int_(np.floor(point_cloud[:, 1] / range_y * (bev_width - 1)) + bev_width / 2)

        # sort-3times
        indices = np.lexsort((-point_cloud[:, 2], point_cloud[:, 1], point_cloud[:, 0]))
        point_cloud = point_cloud[indices]

        # Height Map
        height_map = np.zeros((bev_height, bev_width))

        _, indices = np.unique(point_cloud[:, 0:2], axis=0, return_index=True)
        point_cloud_frac = point_cloud[indices]

        # some important problem is image coordinate is (y,x), not (x,y)
        max_height = float(np.abs(data_params['z_max'] - data_params['z_min']))
        height_map[np.int_(point_cloud_frac[:, 0]), np.int_(point_cloud_frac[:, 1])] = point_cloud_frac[:, 2] / max_height

        # Intensity Map & DensityMap
        intensity_map = np.zeros((bev_height, bev_width))
        density_map = np.zeros((bev_height, bev_width))

        _, indices, counts = np.unique(point_cloud[:, 0:2],
                                       axis=0,
                                       return_index=True,
                                       return_counts=True)

        point_cloud_top = point_cloud[indices]
        normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
        intensity_map[np.int_(point_cloud_top[:, 0]), np.int_(point_cloud_top[:, 1])] = point_cloud_top[:, 3]
        density_map[np.int_(point_cloud_top[:, 0]), np.int_(point_cloud_top[:, 1])] = normalized_counts

        rgb_map = np.zeros((bev_height, bev_width, 3))
        rgb_map[:, :, 0] = density_map      # r_map
        rgb_map[:, :, 1] = height_map       # g_map
        rgb_map[:, :, 2] = intensity_map    # b_map

        return rgb_map

    def filter_roi(self, pts):
        mask = np.where((pts[:, 0] >= self.x_min) & (pts[:, 0] <= self.x_max) &
                        (pts[:, 1] >= self.y_min) & (pts[:, 1] <= self.y_max) &
                        (pts[:, 2] >= self.z_min) & (pts[:, 2] <= self.z_max))
        pts = pts[mask]

        return pts

    def encode(self, labels):
        """
        Encode the label to match the model output format
        param labels (array): class, x, y, w, h, angle
        param anchors (array): anchors
        return: encoded label
        """
        range_x = data_params['x_max'] - data_params['x_min']
        range_y = data_params['y_max'] - data_params['y_min']

        anchors_on_image = np.array([self.image_width, self.image_height]) * anchors / np.array([range_y, range_x])
        n_anchors = np.shape(anchors_on_image)[0]

        encoded_labels = np.zeros([self.grid_height, self.grid_width, n_anchors, (6 + 1 + 1)], dtype=np.float32)
        for i in range(labels.shape[0]):
            rois = labels[i][1:]
            classes = np.array(labels[i][0], dtype=np.int32)
            active_indexes = self.get_active_anchors(rois[2:4], anchors_on_image, self.iou_threshold)
            grid_x, grid_y = self.get_grid_cell(rois, self.image_width, self.image_height, self.grid_width, self.grid_height)
            for active_index in active_indexes:
                anchor_label = self.roi2label(rois, anchors_on_image[active_index], self.image_width, self.image_height, self.grid_width, self.grid_height)
                encoded_labels[grid_y, grid_x, active_index] = np.concatenate((anchor_label, [classes], [1.0]))

        return encoded_labels

    def get_active_anchors(self, box_wh, anchors, iou_threshold):
        """
        Get the index of the anchor that matches the ground truth box
        param box_wh (list, tuple):  Width and height of a box
        param anchors (array): anchors
        param iou_th: Match threshold
        return (list):
        """
        index = []
        iou_max, index_max = 0, 0
        for i, a in enumerate(anchors):
            iou = calc_iou_wh(box_wh, a)
            if iou > iou_threshold:
                index.append(i)
            if iou > iou_max:
                iou_max, index_max = iou, i
        if len(index) == 0:
            index.append(index_max)
        return index

    def get_grid_cell(self, roi, img_w, img_h, grid_w, grid_h):  # roi[x, y, w, h, rz]
        """
        Get the grid cell into which the object falls
        param roi : [x, y, w, h, rz]
        param img_w: The width of images
        param img_h: The height of images
        param grid_w:
        param grid_h:
        return (int, int):
        """
        x_center = roi[0]
        y_center = roi[1]
        grid_x = np.minimum(int(grid_w * x_center / img_w), grid_w - 1)
        grid_y = np.minimum(int(grid_h * y_center / img_h), grid_h - 1)
        return grid_x, grid_y

    def roi2label(self, roi, anchor, img_w, img_h, grid_w, grid_h):
        """
        Encode the label to match the model output format
        param roi: x, y, w, h, angle

        return: encoded label
        """
        x_center = roi[0]
        y_center = roi[1]
        w = grid_w * roi[2] / img_w
        h = grid_h * roi[3] / img_h
        anchor_w = grid_w * anchor[0] / img_w
        anchor_h = grid_h * anchor[1] / img_h
        grid_x = grid_w * x_center / img_w
        grid_y = grid_h * y_center / img_h
        grid_x_offset = grid_x - int(grid_x)
        grid_y_offset = grid_y - int(grid_y)
        roi_w_scale = np.log(w / anchor_w + 1e-16)
        roi_h_scale = np.log(h / anchor_h + 1e-16)
        re = np.cos(roi[4])
        im = np.sin(roi[4])
        label = [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale, re, im]
        return label

