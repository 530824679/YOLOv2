# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : tfrecord.py
# Description :create and parse tfrecord
# --------------------------------------

import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import numpy as np
import tensorflow as tf
from data.dataset import Dataset
from cfg.config import path_params, model_params, solver_params, classes_map

class TFRecord(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.tfrecord_dir = path_params['tfrecord_dir']
        self.train_tfrecord_name = path_params['train_tfrecord_name']
        self.input_width = model_params['input_width']
        self.input_height = model_params['input_height']
        self.channels = model_params['channels']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.class_num = len(model_params['classes'])
        self.batch_size = solver_params['batch_size']
        self.dataset = Dataset()

    # 数值形式的数据,首先转换为string,再转换为int形式进行保存
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # 数组形式的数据,首先转换为string,再转换为二进制形式进行保存
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def create_tfrecord(self):
        # 获取作为训练验证集的图片序列
        trainval_path = os.path.join(self.data_path, 'trainval.txt')

        tf_file = os.path.join(self.tfrecord_dir, self.train_tfrecord_name)
        if os.path.exists(tf_file):
            os.remove(tf_file)

        # 循环写入每一帧点云转换的bev和标签到tfrecord文件
        writer = tf.python_io.TFRecordWriter(tf_file)
        with open(trainval_path, 'r') as read:
            lines = read.readlines()
            for line in lines:
                filename = line[0:-1]
                image_raw, bbox_raw = self.dataset.load_data(filename)

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_raw])),
                    }))
                writer.write(example.SerializeToString())
        writer.close()
        print('Finish trainval.tfrecord Done')

    def parse_single_example(self, serialized_example):
        """
        :param serialized_example:待解析的tfrecord文件的名称
        :return: 从文件中解析出的单个样本的相关特征，image, label
        """
        # 解析单个样本文件
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'bbox': tf.FixedLenFeature([], tf.string),
            })

        image = features['image']
        bbox = features['bbox']

        # 进行解码
        tf_image = tf.decode_raw(image, tf.uint8)
        tf_bbox = tf.decode_raw(bbox, tf.float32)

        # 转换为网络输入所要求的形状
        tf_image = tf.reshape(tf_image, [self.input_height, self.input_width, self.channels])
        tf_label = tf.reshape(tf_bbox, [30, 5])

        # 图像和标签相对于图像空间归一化
        tf_image = tf.cast(tf_image, tf.float32) / 255.0
        tf_label = tf_label[..., 0:4] / 416.0

        y_true = tf.py_func(self.dataset.preprocess_true_boxes, inp=[tf_label], Tout=[tf.float32])
        y_true = tf.reshape(y_true, [self.grid_height, self.grid_width, 5, 6])
        return tf_image, y_true

    def create_dataset(self, filenames, batch_size=1, is_shuffle=False):
        """
        :param filenames: record file names
        :param batch_size: batch size
        :param is_shuffle: whether shuffle
        :param n_repeats: number of repeats
        :return:
        """
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_single_example, num_parallel_calls=4)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(batch_size)
        if is_shuffle:
            dataset = dataset.shuffle(buffer_size=20*batch_size)
        dataset = dataset.batch(batch_size)

        return dataset

if __name__ == '__main__':
    tfrecord = TFRecord()
    tfrecord.create_tfrecord()

    # import matplotlib.pyplot as plt
    # file = '/home/chenwei/HDD/Project/YOLOv2/tfrecord/train.tfrecord'
    # tfrecord = TFRecord()
    # dataset = tfrecord.create_dataset(file, batch_size=2, is_shuffle=False)
    # iterator = dataset.make_one_shot_iterator()
    # images, labels = iterator.get_next()
    #
    # with tf.Session() as sess:
    #     for i in range(20):
    #         images_, labels_ = sess.run([images, labels])
    #         print(images_.shape, labels.shape)
    #         for images_i, boxes_ in zip(images_, labels_):
    #             boxes_ = boxes_[..., 0:4] * 416
    #             valid = (np.sum(boxes_, axis=-1) > 0).tolist()
    #             print([int(idx) for idx in boxes_[:, 0][valid].tolist()])
    #             for box in boxes_[:, 0:4][valid].tolist():
    #                 cv2.rectangle(images_i, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    #             cv2.imshow("image", images_i)
    #             cv2.waitKey(0)
