# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset_utils.py
# Description :数据集清理
# --------------------------------------

import os
import numpy as np
from cfg.config import path_params

def create_trainval_txt(root_path):
    data_path = os.path.join(root_path, 'object/training/livox')
    trainval = os.path.join(root_path, 'ImageSets/Main/trainval.txt')

    if os.path.exists(trainval):
        os.remove(trainval)

    file_obj = open(trainval, 'w', encoding='utf-8')
    file_list = os.listdir(data_path)
    for file in file_list:
        filename = os.path.splitext(file)[0]
        file_obj.writelines(filename)
        file_obj.write('\n')
    file_obj.close()


if __name__ == '__main__':
    create_trainval_txt(path_params['data_path'])