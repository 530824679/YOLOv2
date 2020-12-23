# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : augmentation.py
# Description : data augmentation
# --------------------------------------
import cv2
import math
import random
import numpy as np
from PIL import Image

def random_horizontal_flip(image, bboxes):
    """
    Randomly horizontal flip the image and correct the box
    :param image: BGR image data shape is [height, width, channel]
    :param bboxes: bounding box shape is [num, 4]
    :return: result
    """
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

    return image, bboxes

def letterbox_resize(image, target_size, bboxes, interp=0):
    """
    Resize the image and correct the bbox accordingly.
    :param image: BGR image data shape is [height, width, channel]
    :param bboxes: bounding box shape is [num, 4]
    :param target_size: input size
    :param interp:
    :return: result
    """
    origin_height, origin_width = image.shape[:2]
    input_height, input_width = target_size

    resize_ratio = min(input_width / origin_width, input_height / origin_height)
    resize_width = int(resize_ratio * origin_width)
    resize_height = int(resize_ratio * origin_height)

    image_resized = cv2.resize(image, (resize_width, resize_height), interpolation=interp)
    image_padded = np.full((input_height, input_width, 3), 128, np.uint8)

    dw = int((input_width - resize_width) / 2)
    dh = int((input_height - resize_height) / 2)

    image_padded[dh:resize_height + dh, dw:resize_width + dw, :] = image_resized

    if bboxes is None:
        return image_padded
    else:
        # xmin, xmax, ymin, ymax
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
        return image_padded, bboxes

def random_vertical_flip(image, bboxes):
    """
    Randomly vertical flip the image and correct the box
    :param image: BGR image data shape is [height, width, channel]
    :param bboxes: bounding box shape is [num, 4]
    :return: result
    """
    if random.random() < 0.5:
        h, _, _ = image.shape
        image = image[::-1, :, :]
        bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]

    return image, bboxes

def random_expand(image, bboxes, max_ratio=3, fill=0, keep_ratio=True):
    """
    Random expand original image with borders, this is identical to placing
    the original image on a larger canvas.
    :param image: BGR image data shape is [height, width, channel]
    :param bboxes: bounding box shape is [num, 4]
    :param max_ratio: Maximum ratio of the output image on both direction(vertical and horizontal)
    :param fill: The value(s) for padded borders.
    :param keep_ratio: If `True`, will keep output image the same aspect ratio as input.
    :return: result
    """
    h, w, c = image.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    dst = np.full(shape=(oh, ow, c), fill_value=fill, dtype=image.dtype)

    dst[off_y:off_y + h, off_x:off_x + w, :] = image

    # correct bbox
    bboxes[:, :2] += (off_x, off_y)
    bboxes[:, 2:4] += (off_x, off_y)

    return dst, bboxes

def random_color_distort(image, brightness=32, hue=18, saturation=0.5, value=0.5):
    """
    randomly distort image color include brightness, hue, saturation, value.
    :param image: BGR image data shape is [height, width, channel]
    :param brightness:
    :param hue:
    :param saturation:
    :param value:
    :return: result
    """
    def random_hue(image_hsv, hue):
        if random.random() < 0.5:
            hue_delta = np.random.randint(-hue, hue)
            image_hsv[:, :, 0] = (image_hsv[:, :, 0] + hue_delta) % 180
        return image_hsv

    def random_saturation(image_hsv, saturation):
        if random.random() < 0.5:
            saturation_mult = 1 + np.random.uniform(-saturation, saturation)
            image_hsv[:, :, 1] *= saturation_mult
        return image_hsv

    def random_value(image_hsv, value):
        if random.random() < 0.5:
            value_mult = 1 + np.random.uniform(-value, value)
            image_hsv[:, :, 2] *= value_mult
        return image_hsv

    def random_brightness(image, brightness):
        if random.random() < 0.5:
            image = image.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness, brightness))
            image = image + brightness_delta
        return np.clip(image, 0, 255)

    # brightness
    image = random_brightness(image, brightness)
    image = image.astype(np.uint8)

    # color jitter
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        image_hsv = random_value(image_hsv, value)
        image_hsv = random_saturation(image_hsv, saturation)
        image_hsv = random_hue(image_hsv, hue)
    else:
        image_hsv = random_saturation(image_hsv, saturation)
        image_hsv = random_hue(image_hsv, hue)
        image_hsv = random_value(image_hsv, value)

    image_hsv = np.clip(image_hsv, 0, 255)
    image = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return image

def mix_up(image_1, image_2, bbox_1, bbox_2):
    """
    Overlay images and tags
    :param image_1: BGR image_1 data shape is [height, width, channel]
    :param image_2: BGR image_2 data shape is [height, width, channel]
    :param bbox_1: bounding box_1 shape is [num, 4]
    :param bbox_2: bounding box_2 shape is [num, 4]
    :return:
    """
    height = max(image_1.shape[0], image_2.shape[0])
    width = max(image_1.shape[1], image_2.shape[1])

    mix_image = np.zeros(shape=(height, width, 3), dtype='float32')

    rand_num = np.random.beta(1.5, 1.5)
    rand_num = max(0, min(1, rand_num))

    mix_image[:image_1.shape[0], :image_1.shape[1], :] = image_1.astype('float32') * rand_num
    mix_image[:image_2.shape[0], :image_2.shape[1], :] += image_2.astype('float32') * (1. - rand_num)

    mix_image = mix_image.astype('uint8')

    # the last element of the 2nd dimention is the mix up weight
    bbox_1 = np.concatenate((bbox_1, np.full(shape=(bbox_1.shape[0], 1), fill_value=rand_num)), axis=-1)
    bbox_2 = np.concatenate((bbox_2, np.full(shape=(bbox_2.shape[0], 1), fill_value=1. - rand_num)), axis=-1)
    mix_bbox = np.concatenate((bbox_1, bbox_2), axis=0)
    mix_bbox = mix_bbox.astype(np.int32)

    return mix_image, mix_bbox

def random_crop(image, bboxes):
    """
    Randomly crop the image and correct the box
    :param image: BGR image data shape is [height, width, channel]
    :param bboxes: bounding box shape is [num, 4]
    :return: result
    """
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes

def random_translate(image, bboxes):
    """
    translation image and bboxes
    :param image: BGR image data shape is [height, width, channel]
    :param bbox: bounding box_1 shape is [num, 4]
    :return: result
    """
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

    return image, bboxes

def random_noise(image):
    """
    add noise into image
    :param image: BGR image data shape is [height, width, channel]
    :return: result
    """
    shape = image.shape
    noise = np.random.normal(size=(shape[0], shape[1]))

    out = np.zeros_like(image)
    for i in range(3):
        out[:, :, i] = image[:, :, i]+noise
    out[out > 255] = 255
    out[out < 0] = 0
    out = out.astype('uint8')
    return out

def random_cutout(image, hole_num=2, max_size=(100, 100), min_size=(20, 20), fill_value_mode='zero'):
    """
    cut out mask into image
    :param image: BGR image data shape is [height, width, channel]
    :return: result
    """
    if random.random() < 0.5:
        height, width, _ = image.shape

        if fill_value_mode == 'zero':
            f = np.zeros
            param = {'shape': (height, width, 3)}
        elif fill_value_mode == 'one':
            f = np.one
            param = {'shape': (height, width, 3)}
        else:
            f = np.random.uniform
            param = {'low': 0, 'high': 255, 'size': (height, width, 3)}
        mask = np.ones((height, width, 3), np.int32)

        for index in range(hole_num):
            y = np.random.randint(height)
            x = np.random.randint(width)

            h = np.random.randint(min_size[0], max_size[0] + 1)
            w = np.random.randint(min_size[1], max_size[1] + 1)

            y1 = np.clip(y - h // 2, 0, height)
            y2 = np.clip(y + h // 2, 0, height)
            x1 = np.clip(x - w // 2, 0, width)
            x2 = np.clip(x + w // 2, 0, width)

            mask[y1: y2, x1: x2, :] = 0.

        image = np.where(mask, image, f(**param))

    return np.uint8(image)

def random_rotate(image, bboxes, angle=5, scale=1.):
    """
    rotate image and bboxes
    :param image: BGR image data shape is [height, width, channel]
    :param bboxes: all bounding box in the image. shape is [x_min, y_min, x_max, y_max]
    :param angle: rotate angle
    :param scale: default is 1
    :return: rotate_image:
             rotate_bboxes:
    """
    if random.random() < 0.5:
        height = image.shape[0]
        width = image.shape[1]

        # rotate image
        rangle = np.deg2rad(angle)
        new_width = (abs(np.sin(rangle) * height) + abs(np.cos(rangle) * width)) * scale
        new_height = (abs(np.cos(rangle) * height) + abs(np.sin(rangle) * width)) * scale

        rot_mat = cv2.getRotationMatrix2D((new_width * 0.5, new_height * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(new_width-width)*0.5, (new_height-height)*0.5,0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # warpAffine
        rot_image = cv2.warpAffine(image, rot_mat, (int(math.ceil(new_width)), int(math.ceil(new_height))), flags=cv2.INTER_LANCZOS4)

        # rotate bboxes
        rot_bboxes = list()

        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])
        return rot_image, rot_bboxes
    else:
        return image, bboxes

def random_erasing(image, s_min=0.02, s_max=0.4, ratio=0.3):
    """
    rotate image and bboxes
    :param image: BGR image data shape is [height, width, channel]
    :param s_min: min erasing area region
    :param s_max: max erasing area region
    :param ratio: min aspect ratio range of earsing region
    :return: result
    """
    assert len(image.shape) == 3, 'image should be a 3 dimension numpy array'
    if random.random() < 0.5:
        while True:
            s = (s_min, s_max)
            r = (ratio, 1 / ratio)

            Se = random.uniform(*s) * image.shape[0] * image.shape[1]
            re = random.uniform(*r)

            He = int(round(math.sqrt(Se * re)))
            We = int(round(math.sqrt(Se / re)))

            xe = random.randint(0, image.shape[1])
            ye = random.randint(0, image.shape[0])

            if xe + We <= image.shape[1] and ye + He <= image.shape[0]:
                image[ye: ye + He, xe: xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, image.shape[2]))
                return image

    return image

def random_gridmask(image, mode=1, rotate=1, r_ratio=0.5, d_ratio=1):
    """
    rotate image and bboxes
    :param image: BGR image data shape is [height, width, channel]
    :param mode:
    :param rotate:
    :param r_ratio:
    :param d_ratio:
    :return: result
    """
    if random.random() < 0.5:
        h = image.shape[0]
        w = image.shape[1]
        d1 = 2
        d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(d1, d2)

        if rotate == 1:
            l = np.random.randint(1, d)
        else:
            l = min(max(int(d * r_ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        for i in range(hh // d):
            s = d * i + st_h
            t = min(s + l, hh)
            mask[s:t, :] *= 0
        for i in range(ww // d):
            s = d * i + st_w
            t = min(s + l, ww)
            mask[:, s:t] *= 0

        r = np.random.randint(rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        #  mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        if mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask.astype(np.uint8), axis=2)
        mask = np.tile(mask, [1, 1, 3])

        image = image * mask
    return image

def rand_bbox(shape, lam):
    height = shape[0]
    width = shape[1]
    cut_ratio = np.sqrt(1. - lam)
    cut_height = np.int(height * cut_ratio)
    cut_width = np.int(width * cut_ratio)

    # uniform
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_width // 2, 0, width)
    bby1 = np.clip(cy - cut_height // 2, 0, height)
    bbx2 = np.clip(cx + cut_width // 2, 0, width)
    bby2 = np.clip(cy + cut_height // 2, 0, height)

    return bbx1, bby1, bbx2, bby2

def cut_mix(image_1, image_2, bboxes_1, bboxes_2, beta=1.0):
    # use uniform dist
    lam = np.random.beta(beta, beta)

    image_cutmix = image_1.copy()
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_cutmix.shape, lam)
    image_cutmix[bby1:bby2, bbx1:bbx2, :] = image_2[bby1:bby2, bbx1:bbx2, :]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_1.shape[0] * image_1.shape[1]))

    for i in range(len(bboxes_1)):
        if bboxes_1[i, 0] > bbx1 and bboxes_1[i, 0] < bbx2 and bboxes_1[i, 1] > bby1 and bboxes_1[i, 1] < bby2 and \
                bboxes_1[i, 2] > bbx1 and bboxes_1[i, 2] < bbx2 and bboxes_1[i, 3] > bby1 and bboxes_1[i, 3] > bby2:
            bboxes_1[i, 0] = np.maximum(bbx1, bboxes_1[i, 0])
            bboxes_1[i, 1] = np.maximum(bby2, bboxes_1[i, 1])
            bboxes_1[i, 2] = np.minimum(bbx2, bboxes_1[i, 2])
            bboxes_1[i, 3] = np.maximum(bby1, bboxes_1[i, 3])
        elif bboxes_1[i, 0] > bbx1 and bboxes_1[i, 0] < bbx2 and bboxes_1[i, 1] < bby1 and bboxes_1[i, 1] < bby2 and \
                bboxes_1[i, 2] > bbx1 and bboxes_1[i, 2] < bbx2 and bboxes_1[i, 3] > bby1 and bboxes_1[i, 3] < bby2:
            bboxes_1[i, 0] = np.maximum(bbx1, bboxes_1[i, 0])
            bboxes_1[i, 1] = np.minimum(bby1, bboxes_1[i, 1])
            bboxes_1[i, 2] = np.minimum(bbx2, bboxes_1[i, 2])
            bboxes_1[i, 3] = np.minimum(bby1, bboxes_1[i, 3])
        elif bboxes_1[i, 0] > bbx1 and bboxes_1[i, 0] < bbx2 and bboxes_1[i, 1] > bby1 and bboxes_1[i, 1] < bby2 and \
             bboxes_1[i, 2] > bbx1 and bboxes_1[i, 2] > bbx2 and bboxes_1[i, 3] > bby1 and bboxes_1[i, 3] < bby2:
            bboxes_1[i, 0] = np.maximum(bbx2, bboxes_1[i, 0])
            bboxes_1[i, 1] = np.maximum(bby1, bboxes_1[i, 1])
            bboxes_1[i, 2] = np.maximum(bbx2, bboxes_1[i, 2])
            bboxes_1[i, 3] = np.minimum(bby2, bboxes_1[i, 3])
        elif bboxes_1[i, 0] < bbx1 and bboxes_1[i, 0] < bbx2 and bboxes_1[i, 1] > bby1 and bboxes_1[i, 1] < bby2 and \
             bboxes_1[i, 2] > bbx1 and bboxes_1[i, 2] < bbx2 and bboxes_1[i, 3] > bby1 and bboxes_1[i, 3] < bby2:
            bboxes_1[i, 0] = np.minimum(bbx1, bboxes_1[i, 0])
            bboxes_1[i, 1] = np.maximum(bby1, bboxes_1[i, 1])
            bboxes_1[i, 2] = np.minimum(bbx1, bboxes_1[i, 2])
            bboxes_1[i, 3] = np.minimum(bby2, bboxes_1[i, 3])
        elif bboxes_1[i, 0] > bbx1 and bboxes_1[i, 0] < bbx2 and bboxes_1[i, 1] > bby1 and bboxes_1[i, 1] < bby2 and \
                bboxes_1[i, 2] > bbx1 and bboxes_1[i, 2] < bbx2 and bboxes_1[i, 3] > bby1 and bboxes_1[i, 3] < bby2:
            bboxes_1[i, 0] = 0
            bboxes_1[i, 1] = 0
            bboxes_1[i, 2] = 0
            bboxes_1[i, 3] = 0

    for i in range(len(bboxes_2)):
        bboxes_2[i, 0] = np.maximum(bbx1, bboxes_2[i, 0])
        bboxes_2[i, 1] = np.maximum(bby1, bboxes_2[i, 1])
        bboxes_2[i, 2] = np.minimum(bbx2, bboxes_2[i, 2])
        bboxes_2[i, 3] = np.minimum(bby2, bboxes_2[i, 3])
        if (bboxes_2[i, 0] > bboxes_2[i, 2]) or (bboxes_2[i, 1] > bboxes_2[i, 3]):
            bboxes_2[i, 0] = 0
            bboxes_2[i, 1] = 0
            bboxes_2[i, 2] = 0
            bboxes_2[i, 3] = 0

    bboxes_cutmix = np.concatenate([bboxes_1, bboxes_2], axis=0)

    # compute output
    # loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    return image_cutmix, bboxes_cutmix

def random_affine(image, bboxes, degrees=10, translate=.1, scale=.1, shear=10, border=(0, 0)):
    height = image.shape[0] + border[0] * 2
    width = image.shape[1] + border[1] * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(image.shape[1] / 2, image.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * image.shape[1] + border[1]  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * image.shape[0] + border[0]  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        image = cv2.warpAffine(image, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(bboxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)

        bboxes = bboxes[i]
        bboxes[:, 0:4] = xy[i]

    return image, bboxes

if __name__ == '__main__':
    image_1 = cv2.imread("C:\\Users\\chenw\\Desktop\\test.jpg")
    bboxes_1 = np.array([[355, 66, 484, 301], [425, 296, 502, 320]])
    image_2 = cv2.imread("C:\\Users\\chenw\\Desktop\\test1.jpg")
    bboxes_2 = np.array([[266, 196, 353, 333]])

    image, bboxes = cut_mix(image_1, image_2, bboxes_1, bboxes_2)
    image_copy = image.copy()

    for i, box in enumerate(bboxes):
        cv2.rectangle(image_copy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    cv2.imshow('test', image_copy)
    cv2.waitKey(0)