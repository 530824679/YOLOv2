# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : generator_anchors.py
# Description :generator k anchors
# --------------------------------------

import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import xml.etree.ElementTree as ET
import numpy as np
import glob
import matplotlib.pyplot as plt

def calc_iou(box, clusters):
    """
    calculate iou of a ground truth and all anchor
    :param box: width and height of a ground truth
    :param clusters: numpy shape is (k, 2), k is number of the anchor
    :return: iou of the ground truth and each anchor
    """
    w = np.minimum(clusters[:, 0], box[0])
    h = np.minimum(clusters[:, 1], box[1])

    if np.count_nonzero(w == 0) > 0 or np.count_nonzero(h == 0) > 0:
        raise ValueError("Box has no area")

    intersection = w * h
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou = intersection / (box_area + cluster_area - intersection) * 1.0
    return iou

def avg_iou(boxes, clusters):
    """
    calculate iou mean between ground truth and k anchors
    :param boxes: ground truth
    :param clusters: k clustering center
    :return: average iou
    """
    return np.mean([np.max(calc_iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def kmeans(boxes, k):
    """
    generate k anchors throught clustering
    :param boxes: shape is (n, 2) ground truth, n is the number of ground truth
    :param k: the number of anchors
    :return: shape is (k, 2) anchors, k is the number of anchors
    """
    num = boxes.shape[0]
    distances = np.empty((num, k))
    last_clusters = np.zeros((num,))

    np.random.seed()
    # randomly initialize k clustering center from num ground truth
    clusters = boxes[np.random.choice(num, k, replace=False)]

    while True:
        # calculate the distance of each ground truth and k anchors
        for i in range(num):
            distances[i] = 1 - calc_iou(boxes[i], clusters)
        # select nearest anchor index for each ground truth
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # update clustering center using medium of every group cluster
        for cluster in range(k):
            clusters[cluster] = np.median((boxes[nearest_clusters == cluster]), axis=0)
        last_clusters = nearest_clusters

    return clusters, nearest_clusters

def load_xml(path):
    """
    Gets the width and height of the annotation from dataset
    :param path: path of the dataset
    :return: list width and height
    """
    list_wh = []

    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        height = int(root.findtext("size/height"))
        width = int(root.findtext("size/width"))

        objects = root.findall('object')
        for object in objects:
            bndbox = object.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            width_norm = (xmax - xmin) / width
            height_norm = (ymax - ymin) / height

            list_wh.append([width_norm, height_norm])
    return np.array(list_wh)

def load_txt(root, shape=(416, 416)):
    """
    Gets the width and height of the annotation from dataset
    :param path: path of the dataset
    :param shape: input height and width
    :return: list width and height
    """
    list_wh = []

    labels_path = os.path.join(root, 'labels')
    for txt_file in os.listdir(labels_path):
        filename = os.path.splitext(txt_file)[0]
        image_path = os.path.join(root, 'images', filename + '.jpg')
        image = cv2.imread(image_path)
        image_width = image.shape[1]
        image_height = image.shape[0]

        resize_ratio = min(shape[0] / image_height, shape[1] / image_width)

        f = open(os.path.join(labels_path, txt_file))
        for line in f.readlines():
            data = line.split(' ')
            width = int(float(data[3]) * resize_ratio)
            height = int(float(data[4]) * resize_ratio)

            list_wh.append([width, height])
    return np.array(list_wh)

def plot_data(data, out, k, index):
    for i in range(k):
        color = ['orange', 'green', 'blue', 'gray', 'yellow', 'purple', 'pink', 'black', 'brown']
        mark = ['o', 's', '^', '*', 'd', '+', 'x', 'p', '|']
        lab = 'cluster' + str(i + 1)
        plt.scatter(data[index == i, 0], data[index == i, 1], s=10, c=color[i], marker=mark[i], label=lab)
        # draw the centers
    plt.scatter(out[:, 0], out[:, 1], s=250, marker="*", c="red", label="cluster center")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    path = 'D:\\BaiduNetdiskDownload\\VOC2028\\VOC2028\\Annotations'
    clusters = 5

    #data = load_xml(path)
    data = load_txt('/home/chenwei/HDD/Project/datasets/object_detection/FDDB2016/convert')
    out, cluster_index = kmeans(data, clusters)

    print('Boxes:')
    print(np.array(out))
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Before Sort Ratios:\n {}".format(ratios))
    print("After Sort Ratios:\n {}".format(sorted(ratios)))

    plot_data(data, out, clusters, cluster_index)