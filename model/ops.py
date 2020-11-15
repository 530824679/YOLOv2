# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : ops.py
# Description :base operators.
# --------------------------------------

import tensorflow as tf

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1, name='leaky_relu')

def conv2d(inputs, filters_num, filters_size, pad_size=0, stride=1, batch_normalize=True, activation=leaky_relu, use_bias=False, name='conv2d'):
    if pad_size > 0:
        inputs = tf.pad(inputs, [[0,0], [pad_size, pad_size], [pad_size, pad_size],[0,0]])

    out = tf.layers.conv2d(inputs, filters=filters_num, kernel_size=filters_size, strides=stride, padding='VALID', activation=None, use_bias=use_bias, name=name)

    if batch_normalize:
        out = tf.layers.batch_normalization(out, axis=-1, momentum=0.9, training=False, name=name+'_bn')

    if activation:
        out = activation(out)

    return out

def maxpool(inputs, size=2, stride=2, name='maxpool'):
    with tf.name_scope(name):
         out = tf.layers.max_pooling2d(inputs, pool_size=size, strides=stride, padding='SAME')
    return out

def reorg(inputs, stride):
    return tf.space_to_depth(inputs, block_size=stride)