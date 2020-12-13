# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : network.py
# Description :YOLO v2 network architecture
# --------------------------------------

import numpy as np
import tensorflow as tf
from cfg.config import model_params, solver_params
from model.ops import *

class Network(object):
    def __init__(self, is_train):
        self.is_train = is_train
        self.class_num = len(model_params['classes'])
        self.input_height = model_params['input_height']
        self.input_width = model_params['input_width']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.anchors = model_params['anchors']
        self.anchors_num = len(self.anchors)
        self.output_size = self.anchors_num * (5 + self.class_num)
        self.num_anchors = len(model_params['anchors'])

        self.batch_size = solver_params['batch_size']
        self.num_class = len(model_params['classes'])
        self.iou_threshold = model_params['iou_threshold']
        self.class_scale = model_params['class_scale']
        self.object_scale = model_params['object_scale']
        self.noobject_scale = model_params['noobject_scale']
        self.coord_scale = model_params['coord_scale']

    def build_network(self, inputs, scope='yolo_v2'):
        """
        定义前向传播过程
        :param inputs:待输入的样本图片
        :param scope: 命名空间
        :return: 网络最终的输出
        """
        net = conv2d(inputs, filters_num=32, filters_size=3, pad_size=1, is_train=self.is_train, name='conv1')
        net = maxpool(net, size=2, stride=2, name='pool1')

        net = conv2d(net, 64, 3, 1, is_train=self.is_train, name='conv2')
        net = maxpool(net, 2, 2, name='pool2')

        net = conv2d(net, 128, 3, 1, is_train=self.is_train, name='conv3_1')
        net = conv2d(net, 64, 1, 0, is_train=self.is_train, name='conv3_2')
        net = conv2d(net, 128, 3, 1, is_train=self.is_train, name='conv3_3')
        net = maxpool(net, 2, 2, name='pool3')

        net = conv2d(net, 256, 3, 1, is_train=self.is_train, name='conv4_1')
        net = conv2d(net, 128, 1, 0, is_train=self.is_train, name='conv4_2')
        net = conv2d(net, 256, 3, 1, is_train=self.is_train, name='conv4_3')
        net = maxpool(net, 2, 2, name='pool4')

        net = conv2d(net, 512, 3, 1, is_train=self.is_train, name='conv5_1')
        net = conv2d(net, 256, 1, 0, is_train=self.is_train, name='conv5_2')
        net = conv2d(net, 512, 3, 1, is_train=self.is_train, name='conv5_3')
        net = conv2d(net, 256, 1, 0, is_train=self.is_train, name='conv5_4')
        net = conv2d(net, 512, 3, 1, is_train=self.is_train, name='conv5_5')

        # 存储这一层特征图，以便后面passthrough层
        shortcut = net
        net = maxpool(net, 2, 2, name='pool5')

        net = conv2d(net, 1024, 3, 1, is_train=self.is_train, name='conv6_1')
        net = conv2d(net, 512, 1, 0, is_train=self.is_train, name='conv6_2')
        net = conv2d(net, 1024, 3, 1, is_train=self.is_train, name='conv6_3')
        net = conv2d(net, 512, 1, 0, is_train=self.is_train, name='conv6_4')
        net = conv2d(net, 1024, 3, 1, is_train=self.is_train, name='conv6_5')

        net = conv2d(net, 1024, 3, 1, is_train=self.is_train, name='conv7_1')
        net = conv2d(net, 1024, 3, 1, is_train=self.is_train, name='conv7_2')

        # shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行passthrough处理
        # 这样26*26*512 -> 26*26*64 -> 13*13*256的特征图
        shortcut = conv2d(shortcut, 64, 1, 0, is_train=self.is_train, name='conv_shortcut')
        shortcut = reorg(shortcut, 2)
        net = tf.concat([shortcut, net], axis=-1)
        net = conv2d(net, 1024, 3, 1, is_train=self.is_train, name='conv8')

        # 用一个1*1卷积去调整channel,该层没有bn层和激活函数
        logits = conv2d(net, filters_num=self.output_size, filters_size=1, batch_normalize=False, activation=None, use_bias=True, name='conv_dec')

        return logits

    def reorg_layer(self, feature_maps, anchors=None):
        """
        解码网络输出的特征图
        :param feature_maps:网络输出的特征图
        :param anchors:
        :return: 网络最终的输出
        """
        feature_shape = tf.shape(feature_maps)[1:3]
        ratio = tf.cast([self.input_height, self.input_width] / feature_shape, tf.float32)
        # 将传入的anchors转变成tf格式的常量列表
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        # 网络输出转化——偏移量、置信度、类别概率
        feature_maps = tf.reshape(feature_maps, [-1, feature_shape[0] * feature_shape[1], self.num_anchors, self.class_num + 5])
        # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
        xy_offset = tf.nn.sigmoid(feature_maps[:, :, :, 0:2])
        # 相对于anchor的wh比例，通过e指数解码
        wh_offset = tf.clip_by_value(tf.exp(feature_maps[:, :, :, 2:4]), 1e-9, 50)
        # 置信度，sigmoid函数归一化到0-1
        obj_probs = tf.nn.sigmoid(feature_maps[:, :, :, 4:5])
        # 网络回归的是得分,用softmax转变成类别概率
        class_probs = tf.nn.softmax(feature_maps[:, :, :, 5:])

        # 构建特征图每个cell的左上角的xy坐标
        height_index = tf.range(tf.cast(feature_shape[0], tf.float32), dtype=tf.float32)  # range(0,13)
        width_index = tf.range(tf.cast(feature_shape[1], tf.float32), dtype=tf.float32)  # range(0,13)
        # 变成x_cell=[[0,1,...,12],...,[0,1,...,12]]和y_cell=[[0,0,...,0],[1,...,1]...,[12,...,12]]
        x_cell, y_cell = tf.meshgrid(height_index, width_index)

        x_cell = tf.reshape(x_cell, [1, -1, 1])  # 和上面[H*W,num_anchors,num_class+5]对应
        y_cell = tf.reshape(y_cell, [1, -1, 1])
        xy_cell = tf.stack([x_cell, y_cell], axis=-1)

        bboxes_xy = (xy_cell + xy_offset) * ratio[::-1]
        bboxes_wh = (rescaled_anchors * wh_offset) * ratio[::-1]

        if self.is_train == False:
            bboxes_xywh = tf.concat([bboxes_xy, bboxes_wh], axis=-1)
            bboxes_corners = tf.stack([bboxes_xywh[..., 0] - bboxes_xywh[..., 2] / 2,
                               bboxes_xywh[..., 1] - bboxes_xywh[..., 3] / 2,
                               bboxes_xywh[..., 0] + bboxes_xywh[..., 2] / 2,
                               bboxes_xywh[..., 1] + bboxes_xywh[..., 3] / 2], axis=3)
            return bboxes_corners, obj_probs, class_probs

        bboxes_xy = tf.reshape(bboxes_xy, [-1, feature_shape[0], feature_shape[1], self.num_anchors, 2])
        bboxes_wh = tf.reshape(bboxes_wh, [-1, feature_shape[0], feature_shape[1], self.num_anchors, 2])

        return bboxes_xy, bboxes_wh

    def calc_loss(self, logits, y_true):
        feature_size = tf.shape(logits)[1:3]

        ratio = tf.cast([self.input_height, self.input_width] / feature_size, tf.float32)

        # ground truth
        object_coords = y_true[:, :, :, :, 0:4]
        object_masks = y_true[:, :, :, :, 4:5]
        object_probs = y_true[:, :, :, :, 5:]

        # shape: [N, 13, 13, 5, 4] & [N, 13, 13, 5] ==> [V, 4]
        valid_true_boxes = tf.boolean_mask(object_coords, tf.cast(object_masks[..., 0], 'bool'))
        # shape: [V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]

        # predicts
        pred_box_xy, pred_box_wh = self.reorg_layer(logits, self.anchors)

        predictions = tf.reshape(logits, [-1, feature_size[0], feature_size[1], self.num_anchors, self.class_num + 5])
        pred_conf_logits = predictions[:, :, :, :, 4:5]
        pred_prob_logits = predictions[:, :, :, :, 5:]

        # calc iou 计算每个pre_boxe与所有true_boxe的交并比.
        # valid_true_box_xx: [V,2]
        # pred_box_xx: [13,13,5,2]
        # shape: [N, 13, 13, 5, V],
        iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)
        # shape : [N,13,13,5]
        best_iou = tf.reduce_max(iou, axis=-1)

        # get_ignore_mask shape: [N,13,13,5,1] 0,1张量
        ignore_mask = tf.expand_dims(tf.cast(best_iou < self.iou_threshold, tf.float32), -1)

        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.input_height, tf.float32)) * (y_true[..., 3:4] / tf.cast(self.input_width, tf.float32))# y_true[..., 2:3] * y_true[..., 3:4]
        pred_xywh = tf.concat([pred_box_xy, pred_box_wh], axis=-1)
        diou = tf.expand_dims(self.bbox_diou(pred_xywh, object_coords), axis=-1)
        diou_loss = object_masks * box_loss_scale * (1 - diou)

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_masks
        conf_neg_mask = (1 - object_masks) * ignore_mask

        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_masks, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_masks, logits=pred_conf_logits)
        conf_loss = conf_loss_pos + conf_loss_neg

        # shape: [N, 13, 13, 3, 1]
        class_loss = object_masks * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_probs, logits=pred_prob_logits)

        diou_loss = tf.reduce_mean(tf.reduce_sum(diou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1, 2, 3, 4]))

        total_loss = diou_loss + conf_loss + class_loss
        return total_loss, diou_loss, conf_loss, class_loss

    def bbox_diou(self, boxes_1, boxes_2):
        """
        calculate regression loss using diou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # calculate center distance
        center_distance = tf.reduce_sum(tf.square(boxes_1[..., :2] - boxes_2[..., :2]), axis=-1)

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

        # calculate IoU, add epsilon in denominator to avoid dividing by 0
        iou = inter_area / (union_area + tf.keras.backend.epsilon())

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate enclosed diagonal distance
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

        # calculate diou add epsilon in denominator to avoid dividing by 0
        diou = iou - 1.0 * center_distance / (enclose_diagonal + tf.keras.backend.epsilon())

        return diou

    def broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        # shape:
        # true_box_??: [V, 2] V:目标数量
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2] , 扩张维度方便进行维度广播
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2] V:该尺度下分feature_map 下所有的目标是目标数量
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] --> [N, 13, 13, 3, V, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2] 维度广播
        # 真boxe,左上角,右下角, 假boxe的左上角,右小角,
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / + 2.,  # 取最靠右的左上角
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,  # 取最靠左的右下角
                                    true_box_xy + true_box_wh / 2.)
        # tf.maximun 去除那些没有面积交叉的矩形框, 置0
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)  # 得到重合区域的长和宽

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # 重合部分面积
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]  # 预测区域面积
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]  # 真实区域面积
        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + tf.keras.backend.epsilon())

        return iou