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
from cfg.config import model_params
from model.ops import *

class Network(object):
    def __init__(self, is_train):
        self.class_num = len(model_params['classes'])
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.anchor_num = model_params['anchor_num']
        self.output_size = self.anchor_num * (5 + self.class_num)
        self.anchors = model_params['anchors']

    def forward(self, inputs):
        feature_maps = self.build_network(inputs)
        output = self.reorg_layer(feature_maps, self.anchors)
        return feature_maps, output

    def build_network(self, inputs, scope='yolo_v2'):
        """
        定义前向传播过程
        :param inputs:待输入的样本图片
        :param scope: 命名空间
        :return: 网络最终的输出
        """
        with tf.name_scope(scope):
            net = conv2d(inputs, filters_num=32, filters_size=3, pad_size=1, name='conv1')
            net = maxpool(net, size=2, stride=2, name='pool1')

            net = conv2d(net, 64, 3, 1, name='conv2')
            net = maxpool(net, 2, 2, name='pool2')

            net = conv2d(net, 128, 3, 1, name='conv3_1')
            net = conv2d(net, 64, 1, 0, name='conv3_2')
            net = conv2d(net, 128, 3, 1, name='conv3_3')
            net = maxpool(net, 2, 2, name='pool3')

            net = conv2d(net, 256, 3, 1, name='conv4_1')
            net = conv2d(net, 128, 1, 0, name='conv4_2')
            net = conv2d(net, 256, 3, 1, name='conv4_3')
            net = maxpool(net, 2, 2, name='pool4')

            net = conv2d(net, 512, 3, 1, name='conv5_1')
            net = conv2d(net, 256, 1, 0, name='conv5_2')
            net = conv2d(net, 512, 3, 1, name='conv5_3')
            net = conv2d(net, 256, 1, 0, name='conv5_4')
            net = conv2d(net, 512, 3, 1, name='conv5_5')

            # 存储这一层特征图，以便后面passthrough层
            shortcut = net
            net = maxpool(net, 2, 2, name='pool5')

            net = conv2d(net, 1024, 3, 1, name='conv6_1')
            net = conv2d(net, 512, 1, 0, name='conv6_2')
            net = conv2d(net, 1024, 3, 1, name='conv6_3')
            net = conv2d(net, 512, 1, 0, name='conv6_4')
            net = conv2d(net, 1024, 3, 1, name='conv6_5')

            net = conv2d(net, 1024, 3, 1, name='conv7_1')
            net = conv2d(net, 1024, 3, 1, name='conv7_2')

            # shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行passthrough处理
            # 这样26*26*512 -> 26*26*64 -> 13*13*256的特征图
            shortcut = conv2d(shortcut, 64, 1, 0, name='conv_shortcut')
            shortcut = reorg(shortcut, 2)
            net = tf.concat([shortcut, net], axis=-1)
            net = conv2d(net, 1024, 3, 1, name='conv8')

            # 用一个1*1卷积去调整channel,该层没有bn层和激活函数
            logits = conv2d(net, filters_num=self.output_size, filters_size=1, batch_normalize=False, activation=None, use_bias=True, name='logits')

        return logits

    def reorg_layer(self, feature_maps, anchors=None):
        """
        解码网络输出的特征图
        :param feature_maps:网络输出的特征图
        :param anchors:
        :return: 网络最终的输出
        """
        feature_shape = tf.shape(feature_maps)[1:3]
        batch_size = tf.shape(feature_maps)[0]
        num_anchors = len(anchors)

        # 将传入的anchors转变成tf格式的常量列表
        anchors = tf.constant(anchors, dtype=tf.float32)

        # 网络输出转化——偏移量、置信度、类别概率
        predict = tf.reshape(feature_maps, [batch_size, feature_shape[0], feature_shape[1], num_anchors, self.class_num + 5])
        # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
        xy_offset = tf.nn.sigmoid(predict[:, :, :, :, 0:2])
        # 相对于anchor的wh比例，通过e指数解码
        wh_offset = tf.clip_by_value(tf.exp(predict[:, :, :, :, 2:4]), 1e-9, 50)
        # 置信度，sigmoid函数归一化到0-1
        obj_probs = tf.nn.sigmoid(predict[:, :, :, :, 4:5])
        # 网络回归的是得分,用softmax转变成类别概率
        class_probs = tf.nn.softmax(predict[:, :, :, :, 5:])

        # 构建特征图每个cell的左上角的xy坐标
        height_index = tf.range(feature_shape[0], dtype=tf.int32)
        width_index = tf.range(feature_shape[1], dtype=tf.int32)
        x_cell, y_cell = tf.meshgrid(height_index, width_index)

        x_cell = tf.reshape(x_cell, [-1, 1])
        y_cell = tf.reshape(y_cell, [-1, 1])
        xy_cell = tf.concat([x_cell, y_cell], axis=-1)
        xy_cell = tf.cast(tf.reshape(xy_cell, [feature_shape[0], feature_shape[1], 1, 2]), tf.float32)

        # decode to raw image norm 0-1
        bboxes_xy = (xy_cell + xy_offset) / tf.cast(feature_shape[::-1], tf.float32)
        bboxes_wh = (anchors * wh_offset) / tf.cast(feature_shape[::-1], tf.float32)
        bboxes_xywh = tf.concat([bboxes_xy, bboxes_wh], axis=-1)

        return tf.concat([bboxes_xywh, obj_probs, class_probs], axis=-1)