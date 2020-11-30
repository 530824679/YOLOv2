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
        self.image_width = model_params['image_width']
        self.image_height = model_params['image_height']
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
                image, bbox = self.dataset.load_data(filename)

                if len(bbox) == 0:
                    continue

                image_string = image.tobytes()
                bbox_string = bbox.tobytes()
                bbox_shape = bbox.shape

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_string])),
                        'bbox_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=bbox_shape))
                    }))
                writer.write(example.SerializeToString())
        writer.close()
        print('Finish trainval.tfrecord Done')

    def parse_single_example(self, tfrecord_file):
        """
        :param file_name:待解析的tfrecord文件的名称
        :return: 从文件中解析出的单个样本的相关特征，image, label
        """
        # 定义解析TFRecord文件操作
        reader = tf.TFRecordReader()

        # 创建样本文件名称队列
        filename_queue = tf.train.string_input_producer([tfrecord_file])

        # 解析单个样本文件
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'bbox': tf.FixedLenFeature([], tf.string),
                'bbox_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64)
            })

        image = features['image']
        bbox = features['bbox']
        bbox_shape = features['bbox_shape']

        return image, bbox, bbox_shape

    def parse_batch_examples(self, file_name):
        """
        :param file_name:待解析的tfrecord文件的名称
        :return: 解析得到的batch_size个样本
        """
        batch_size = self.batch_size
        min_after_dequeue = 100
        num_threads = 2
        capacity = min_after_dequeue + 3 * batch_size


        image, label, bbox_shape = self.parse_single_example(file_name)
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size,
                                                          num_threads=num_threads,
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)

        # 进行解码
        image_batch = tf.decode_raw(image_batch, tf.uint8)
        label_batch = tf.decode_raw(label_batch, tf.float32)

        # 转换为网络输入所要求的形状
        image_batch = tf.reshape(image_batch, [self.batch_size, self.image_height, self.image_width, self.channels])
        label_batch = tf.reshape(label_batch, [self.batch_size, bbox_shape.shape[0], 4 + self.class_num])

        return image_batch, label_batch

if __name__ == '__main__':
    tfrecord = TFRecord()
    #tfrecord.create_tfrecord()

    file = '/home/chenwei/HDD/Project/YOLOv2/tfrecord/train.tfrecord'
    tfrecord = TFRecord()
    batch_image, batch_label = tfrecord.parse_batch_examples(file)
    with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1):
            label = sess.run([batch_label])
            #print(image)
            print(label.astype(np.float32))
            # box = label[0, ]

            #print(np.shape(image), np.shape(label))
            # img = np.array(image[0])
            # cv2.imshow('img', img[0, ...])
            # cv2.waitKey(0)
        # print(type(example))
        coord.request_stop()
        # coord.clear_stop()
        coord.join(threads)