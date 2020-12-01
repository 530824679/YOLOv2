# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : loss.py
# Description :Yolo_v2 Loss损失函数.
# --------------------------------------

import numpy as np
import tensorflow as tf
from cfg.config import model_params, solver_params

class Loss(object):
    def __init__(self):
        """
        :param predicts:网络的输出 anchor_num * (5 + class_num)
        :param labels:标签信息
        :param scope:命名loss
        """
        self.batch_size = solver_params['batch_size']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.num_class = len(model_params['classes'])
        self.anchors = model_params['anchors']
        self.anchors_num = len(self.anchors)
        self.iou_threshold = model_params['iou_threshold']
        self.class_scale = model_params['class_scale']
        self.object_scale = model_params['object_scale']
        self.noobject_scale = model_params['noobject_scale']
        self.coord_scale = model_params['coord_scale']

    def calc_loss(self, pred_feat, pred_bbox, y_true):
        feature_shape = tf.shape(pred_feat)[1:3]
        predicts = tf.reshape(pred_feat, [-1, feature_shape[0], feature_shape[1], 5, (5 + 1)])
        conv_conf = predicts[:, :, :, :, 4:5]
        conv_prob = predicts[:, :, :, :, 5:]

        pred_xywh = pred_bbox[:, :, :, :, 0:4]
        pred_conf = pred_bbox[:, :, :, :, 4:5]
        pred_class = tf.argmax(pred_bbox[:, :, :, :, 5:], axis=-1)

        label_xywh = y_true[:, :, :, :, 0:4]
        object_mask = y_true[:, :, :, :, 4:5]
        label_prob = y_true[:, :, :, :, 5:]
        label_class = tf.argmax(y_true[:, :, :, :, 5:], axis=-1)

        """
        compare online statistics
        """
        true_mins = label_xywh[..., 0:2] - label_xywh[..., 2:4] / 2.
        true_maxs = label_xywh[..., 0:2] + label_xywh[..., 2:4] / 2.
        pred_mins = pred_xywh[..., 0:2] - pred_xywh[..., 2:4] / 2.
        pred_maxs = pred_xywh[..., 0:2] + pred_xywh[..., 2:4] / 2.

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxs = tf.minimum(pred_maxs, true_maxs)

        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_area = label_xywh[..., 2] * label_xywh[..., 3]
        pred_area = pred_xywh[..., 2] * pred_xywh[..., 3]

        union_area = pred_area + true_area - intersect_area
        iou_scores = tf.truediv(intersect_area, union_area)

        # coord loss label_wh normalzation 0-1
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / tf.cast(feature_shape[0], dtype=tf.float32) / tf.cast(feature_shape[1], dtype=tf.float32)
        ciou = tf.expand_dims(self.bbox_iou(pred_xywh, label_xywh), axis=-1)
        ciou_loss = object_mask * bbox_loss_scale * (1 - ciou)

        # confidence loss
        valid_boxes = tf.boolean_mask(label_xywh[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))
        bboxes = tf.concat([valid_boxes[:, 0:2], valid_boxes[:, 2:4]], axis=-1)
        # shape: [V, 2] ——> [1, V, 2]
        bboxes = tf.expand_dims(bboxes, 0)
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        best_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        noobject_mask = (1.0 - object_mask) * tf.cast( best_iou < self.iou_threshold, tf.float32)

        object_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=conv_conf)
        noobject_loss = noobject_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=conv_conf)
        conf_loss = self.object_scale * object_loss + self.noobject_scale * noobject_loss

        # prob loss
        prob_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_prob)

        coord_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        total_loss = coord_loss + conf_loss + prob_loss
        return total_loss, coord_loss, conf_loss, prob_loss

    # def calc_loss(self, logits, predicts, labels):
    #     anchors = tf.constant(self.anchors, dtype=tf.float32)
    #     anchors = tf.reshape(anchors, [1, 1, self.anchors_num, 2])  # 存放输入的anchors的wh
    #
    #     pred_boxes = predicts[:, :, :, :, 0:4]
    #     pred_conf = predicts[:, :, :, :, 4:5]
    #     pred_classes = predicts[:, :, :, :, 5:]
    #
    #     label_boxes = labels[:, :, :, :, 0:4]
    #     label_response = labels[:, :, :, :, 4:5]
    #     label_classes = labels[:, :, :, :, 5:]
    #
    #     iou = self.calc_iou(label_boxes, pred_boxes)
    #     best_box = tf.to_float(tf.equal(iou, tf.reduce_max(iou, axis=-1, keep_dims=True)))
    #     confs = tf.expand_dims(best_box, 4) * label_response
    #
    #     conid = self.noobject_scale * (1.0 - confs) + self.object_scale * confs
    #     cooid = self.coord_scale * confs
    #     proid = self.class_scale * confs
    #
    #     coord_loss = cooid * tf.square(pred_boxes - label_boxes)
    #     conf_loss = conid * tf.square(pred_conf - label_response)
    #     class_loss = proid * tf.square(pred_classes - label_classes)
    #
    #     loss = tf.concat([coord_loss, conf_loss, class_loss], axis=4)
    #     loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]), name='loss')
    #
    #     return loss, coord_loss, conf_loss, class_loss

    def bbox_iou(self, boxes_1, boxes_2):
        """
        calculate regression loss using iou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate iou add epsilon in denominator to avoid dividing by 0
        iou = inter_area / (union_area + tf.keras.backend.epsilon())

        return iou