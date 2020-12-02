# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : configs.py
# Description :config parameters
# --------------------------------------
import os

path_params = {
    'data_path': '/home/chenwei/HDD/Project/datasets/object_detection/FDDB2016/convert',
    'checkpoints_dir': './checkpoints',
    'logs_dir': './logs',
    'tfrecord_dir': '/home/chenwei/HDD/Project/YOLOv2/tfrecord',
    'checkpoints_name': 'model.ckpt',
    'train_tfrecord_name': 'train.tfrecord',
    'test_output_dir': './test'
}

model_params = {
    'input_height': 416,            # 图片高度
    'input_width': 416,             # 图片宽度
    'channels': 3,                  # 输入图片通道数
    'anchors': [[27, 39], [60, 91], [82, 122], [111, 167], [172, 256]],
    'classes': ['face'],
    'grid_height': 13,              # 输出特征图的网格高度
    'grid_width': 13,               # 输出特征图的网格宽度
    'anchor_num': 5,                # 每个网格负责预测的BBox个数
    'object_scale': 1.0,            # 置信度有目标权重
    'noobject_scale': 5.0,          # 置信度无目标权重
    'class_scale': 1.0,             # 分类损失权重
    'coord_scale': 1.0,             # 定位损失权重
    'iou_threshold': 0.6,
}

solver_params = {
    'gpu': '0',                     # 使用的gpu索引
    'learning_rate': 0.01,        # 初始学习率
    'decay_steps': 30000,           #衰变步数
    'decay_rate': 0.1,              #衰变率
    'staircase': True,
    'batch_size': 4,                # 每批次输入的数据个数
    'epoches': 50000,               # 训练的最大迭代次数
    'save_step': 1000,              # 权重保存间隔
    'log_step': 1000,               # 日志保存间隔
    'weight_decay': 0.0001,         # 正则化系数
    'restore': False                # 支持restore
}

test_params = {
    'prob_threshold': 0.01,         # 类别置信度分数阈值
    'iou_threshold': 0.1,           # nms阈值，小于0.4被过滤掉
    'max_output_size': 10           # nms选择的边界框最大数量
}

classes_map = {'face': 0}