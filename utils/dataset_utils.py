# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset_utils.py
# Description :数据集清理
# --------------------------------------

import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
from cfg.config import path_params

def create_trainval_txt(root_path):
    data_path = os.path.join(root_path, 'images')
    trainval = os.path.join(root_path, 'trainval.txt')

    if os.path.exists(trainval):
        os.remove(trainval)

    file_obj = open(trainval, 'w', encoding='utf-8')
    file_list = os.listdir(data_path)
    for file in file_list:
        filename = os.path.splitext(file)[0]
        file_obj.writelines(filename)
        file_obj.write('\n')
    file_obj.close()

def create_fddb_txt():

    annotation_dir = "/home/chenwei/HDD/Project/datasets/object_detection/FDDB2016/FDDB-folds"
    origin_image_dir = "/home/chenwei/HDD/Project/datasets/object_detection/FDDB2016/originalPics"

    images_dir = "/home/chenwei/HDD/Project/datasets/object_detection/FDDB2016/convert/images"
    labels_dir = "/home/chenwei/HDD/Project/datasets/object_detection/FDDB2016/convert/labels"

    if not os.path.exists(annotation_dir):
        os.mkdir(annotation_dir)
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    count = 1
    for i in range(10):
        annotation_path = os.path.join(annotation_dir, "FDDB-fold-%0*d-ellipseList.txt"%(2,i+1))
        annotation_file = open(annotation_path)
        while(True):
            filename = annotation_file.readline()[:-1] + ".jpg"
            if not filename:
                break
            line = annotation_file.readline()
            if not line:
                break
            face_num=(int)(line)
            count += 1

            image = cv2.imread(os.path.join(origin_image_dir, filename))
            filename = filename.replace('/', '_')
            cv2.imwrite(os.path.join(images_dir, filename), image)

            label_path = labels_dir + "/" + filename.replace('/','_')[:-3] + "txt"
            label_file = open(label_path, 'w')

            for k in range(face_num):
                line = annotation_file.readline().strip().split()
                major_axis_radius = (float)(line[0])
                minor_axis_radius = (float)(line[1])
                angle = (float)(line[2])
                center_x = (float)(line[3])
                center_y = (float)(line[4])
                angle = angle / 3.1415926*180
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                cv2.ellipse(mask, ((int)(center_x), (int)(center_y)), ((int)(major_axis_radius), (int)(minor_axis_radius)), angle, 0., 360., (255, 255, 255))
                _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for k in range(len(contours)):
                    r = cv2.boundingRect(contours[k])
                    xcenter = r[0] + r[2] / 2
                    ycenter = r[1] + r[3] / 2

                    labelline = "0" + " " + str(xcenter) + ' ' + str(ycenter) + ' ' + str(r[2]) + ' ' + str(r[3]) + '\n'
                    label_file.write(labelline)
            label_file.close()
    print(count)

if __name__ == '__main__':
    #create_fddb_txt()
    create_trainval_txt('/home/chenwei/HDD/Project/datasets/object_detection/FDDB2016/convert')