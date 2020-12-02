# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : process_utils.py
# Description :function
# --------------------------------------
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import random
import colorsys
import numpy as np


def calc_iou_wh(box1_wh, box2_wh):
    """
    param box1_wh (list, tuple): Width and height of a box
    param box2_wh (list, tuple): Width and height of a box
    return (float): iou
    """
    min_w = min(box1_wh[0], box2_wh[0])
    min_h = min(box1_wh[1], box2_wh[1])
    area_r1 = box1_wh[0] * box1_wh[1]
    area_r2 = box2_wh[0] * box2_wh[1]
    intersect = min_w * min_h
    union = area_r1 + area_r2 - intersect
    return intersect / union

def calculate_iou(box_1, box_2):
    """
    calculate iou
    :param box_1: (x0, y0, x1, y1)
    :param box_2: (x0, y0, x1, y1)
    :return: value of iou
    """
    bboxes1 = np.transpose(box_1)
    bboxes2 = np.transpose(box_2)

    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)

    # 交集面积
    intersection = int_h * int_w
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])  # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])  # bboxes2面积

    # iou=交集/并集
    iou = intersection / (vol1 + vol2 - intersection)

    return iou

def bboxes_cut(bbox_min_max, bboxes):
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_min_max = np.transpose(bbox_min_max)

    # cut the box
    bboxes[0] = np.maximum(bboxes[0],bbox_min_max[0]) # xmin
    bboxes[1] = np.maximum(bboxes[1],bbox_min_max[1]) # ymin
    bboxes[2] = np.minimum(bboxes[2],bbox_min_max[2]) # xmax
    bboxes[3] = np.minimum(bboxes[3],bbox_min_max[3]) # ymax
    bboxes = np.transpose(bboxes)
    return bboxes

def bboxes_sort(classes, scores, bboxes, top_k=400):
    index = np.argsort(-scores)
    classes = classes[index][:top_k]
    scores = scores[index][:top_k]
    bboxes = bboxes[index][:top_k]
    return classes, scores, bboxes

# 计算nms
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = calculate_iou(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]

def preprocess(image, image_size=(416, 416)):
    image_copy = np.copy(image).astype(np.float32)
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    # letter resize
    image_height, image_width = image.shape[:2]
    resize_ratio = min(image_size[0] / image_width, image_size[1] / image_height)
    resize_width = int(resize_ratio * image_width)
    resize_height = int(resize_ratio * image_height)

    image_resized = cv2.resize(image_rgb, (resize_width, resize_height), interpolation=0)
    image_padded = np.full((image_size[0], image_size[1], 3), 128, np.uint8)

    dw = int((image_size[0] - resize_width) / 2)
    dh = int((image_size[1] - resize_height) / 2)

    image_padded[dh:resize_height + dh, dw:resize_width + dw, :] = image_resized

    image_normalized = image_padded.astype(np.float32) / 225.0

    image_expanded = np.expand_dims(image_normalized, axis=0)

    return image_expanded

# 筛选解码后的回归边界框
def postprocess(bboxes, obj_probs, class_probs, image_shape=(416,416), threshold=0.02):
    # boxes shape——> [num, 4]
    bboxes = np.reshape(bboxes, [-1, 4])

    # 将box还原成图片中真实的位置
    bboxes[:, 0:1] *= float(image_shape[1])  # xmin*width
    bboxes[:, 1:2] *= float(image_shape[0])  # ymin*height
    bboxes[:, 2:3] *= float(image_shape[1])  # xmax*width
    bboxes[:, 3:4] *= float(image_shape[0])  # ymax*height
    bboxes = bboxes.astype(np.int32)

    # 将边界框超出整张图片(0,0)—(415,415)的部分cut掉
    bbox_min_max = [0, 0, image_shape[1] - 1, image_shape[0] - 1]
    bboxes = bboxes_cut(bbox_min_max, bboxes)

    # 置信度 * 类别条件概率 = 类别置信度scores
    obj_probs = np.reshape(obj_probs, [-1])
    class_probs = np.reshape(class_probs, [len(obj_probs), -1])
    class_max_index = np.argmax(class_probs, axis=1)
    class_probs = class_probs[np.arange(len(obj_probs)), class_max_index]
    scores = obj_probs * class_probs

    # 类别置信度scores > threshold的边界框bboxes留下
    keep_index = scores > threshold
    class_max_index = class_max_index[keep_index]
    scores = scores[keep_index]
    bboxes = bboxes[keep_index]

    # 排序取前400个
    class_max_index, scores, bboxes = bboxes_sort(class_max_index, scores, bboxes)

    # 计算nms
    class_max_index, scores, bboxes = bboxes_nms(class_max_index, scores, bboxes)

    return bboxes, scores, class_max_index

def boxes_to_corners(boxes):
    box_xy = boxes[..., 0:2]
    box_wh = boxes[..., 2:4]

    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return np.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def visualization(im, bboxes, scores, cls_inds, labels, thr=0.3):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / float(len(labels)), 1., 1.) for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # draw image
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)
        cv2.rectangle(imgcv, (box[0], box[1]), (box[2], box[3]), colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, (255, 255, 255), thick // 3)
    cv2.imshow("test", imgcv)
    cv2.waitKey(0)