# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : loss_utils.py
# Description :Yolo_v2 Loss损失函数.
# --------------------------------------

import numpy as np
import tensorflow as tf
from cfg.config import model_params, solver_params, anchors

class Loss(object):
    def __init__(self, predicts, labels, scope='loss'):
        """
        :param predicts:网络的输出 anchor_num * (5 + class_num)
        :param labels:标签信息
        :param scope:命名loss
        """
        self.batch_size = solver_params['batch_size']
        self.image_size = model_params['image_size']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.num_class = model_params['num_classes']
        self.anchors = anchors
        self.anchors_num = len(self.anchors)
        self.class_scale = model_params['class_scale']
        self.object_scale = model_params['object_scale']
        self.noobject_scale = model_params['noobject_scale']
        self.coord_scale = model_params['coord_scale']
        self.loss_layer(predicts, labels)

    def loss_layer(self, predicts, labels):
        anchors = tf.constant(self.anchors, dtype=tf.float32)
        anchors = tf.reshape(anchors, [1, 1, self.anchors_num, 2])  # 存放输入的anchors的wh

        # # loss不同部分的前面系数
        sprob, sconf, snoob, scoor = scales

        # ground truth [-1, feature_size * feature_size, anchor_num, 4]，真实坐标xywh
        _coords = labels["coords"]

        # class probability [-1, feature_size * feature_size, anchor_num, class_num] ，类别概率
        _probs = labels["probs"]

        # 1 for object, 0 for background, [-1, feature_size * feature_size, anchor_num]，置信度，每个边界框一个
        _confs = labels["confs"]

        # ground truth计算IOU-->_up_left, _down_right
        _wh = tf.pow(_coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
        _areas = _wh[:, :, :, 0] * _wh[:, :, :, 1]
        _centers = _coords[:, :, :, 0:2]
        _up_left, _down_right = _centers - (_wh * 0.5), _centers + (_wh * 0.5)
        # ground truth汇总
        truths = tf.concat([_coords, tf.expand_dims(_confs, -1), _probs], 3)

        # 解码网络输出值
        predicts = tf.reshape(predicts, [-1, self.feature_size, self.feature_size, self.anchors_num, (5 + self.num_class)])

        # 解码预测的定位信息：t_x, t_y, t_w, t_h
        coords = tf.reshape(predicts[:, :, :, :, 0:4], [-1, self.feature_size * self.feature_size, self.anchors_num, 4])
        # 0-1，xy是相对于cell左上角的偏移量
        coords_xy = tf.nn.sigmoid(coords[:, :, :, 0:2])
        # 0-1，除以特征图的尺寸13，解码成相对于整张图片的wh
        coords_wh = tf.sqrt(tf.exp(coords[:, :, :, 2:4]) * anchors / np.reshape([self.feature_size, self.feature_size], [1, 1, 1, 2]))
        # [batch_size, self.feature_size * self.feature_size, B, 4]
        coords = tf.concat([coords_xy, coords_wh], axis=3)

        # 解码预测的置信度
        confs = tf.nn.sigmoid(predicts[:, :, :, :, 4])
        # 每个边界框一个置信度，每个网格有anchor_num个边界框
        confs = tf.reshape(confs, [-1, self.feature_size * self.feature_size, self.anchors_num, 1])

        # 解码预测的类别概率
        probs = tf.nn.softmax(predicts[:, :, :, :, 5:])
        probs = tf.reshape(probs, [-1, self.feature_size * self.feature_size, self.anchors_num, self.num_class])

        # predict汇总 [-1, feature_size * feature_size, anchors_num, (4+1+num_class)]
        preds = tf.concat([coords, confs, probs], axis=3)

        # predict计算iou-->up_left, down_right
        wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([self.feature_size, self.feature_size], [1, 1, 1, 2])
        areas = wh[:, :, :, 0] * wh[:, :, :, 1]
        centers = coords[:, :, :, 0:2]
        up_left, down_right = centers - (wh * 0.5), centers + (wh * 0.5)

        # 计算ground truth和anchor的IOU：
        # 计算IOU只考虑形状，先将anchor与ground truth的中心点都偏移到同一位置（cell左上角），然后计算出对应的IOU值。
        # ①IOU值最大的那个anchor与ground truth匹配，对应的预测框用来预测这个ground truth：计算xywh、置信度c(目标值为1)、类别概率p误差。
        # ②IOU小于某阈值的anchor对应的预测框：只计算置信度c(目标值为0)误差。
        # ③剩下IOU大于某阈值但不是max的anchor对应的预测框：丢弃，不计算任何误差。
        inter_upleft = tf.maximum(up_left, _up_left)
        inter_downright = tf.minimum(down_right, _down_right)
        inter_wh = tf.maximum(inter_downright - inter_upleft, 0.0)
        intersects = inter_wh[:, :, :, 0] * inter_wh[:, :, :, 1]
        ious = tf.truediv(intersects, areas + _areas - intersects)

        best_iou_mask = tf.equal(ious, tf.reduce_max(ious, axis=2, keep_dims=True))
        best_iou_mask = tf.cast(best_iou_mask, tf.float32)
        mask = best_iou_mask * _confs  # [-1, H*W, B]
        mask = tf.expand_dims(mask, -1)  # [-1, H*W, B, 1]

        # 计算各项损失所占的比例权重weight
        confs_w = snoob * (1 - mask) + sconf * mask
        coords_w = scoor * mask
        probs_w = sprob * mask
        weights = tf.concat([coords_w, confs_w, probs_w], axis=3)

        # 计算loss：ground truth汇总和prediction汇总均方差损失函数，再乘以相应的比例权重
        loss = tf.pow(preds - truths, 2) * weights
        loss = tf.reduce_sum(loss, axis=[1, 2, 3])
        loss = 0.5 * tf.reduce_mean(loss)

        return loss
