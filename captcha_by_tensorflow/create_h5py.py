# -*- coding: utf-8 -*-
# @Time : 2021/1/21 11:35
# @Author : xiaojie
# @File : create_h5py.py
# @Software: PyCharm

import os
import numpy as np
import cv2
import h5py
from PIL import Image


def save_image_to_h5py(path):
    """
    生成h5数据集
    :param path: 存储验证码的路径
    :return:
    """

    img_list = []
    label_list = []
    # 直接用文件名做label
    label_text = ""

    for dir_image in os.listdir(path):  # 遍历二级文件每个图片并append信息进数组
        label_text = dir_image.split('.')[0]
        # img = cv2.imdecode(np.fromfile(path + dir_image, dtype=np.uint8), -1) # 可读取中文路径
        img = cv2.imread(os.path.join(path, dir_image))
        img = cv2.resize(img, (122, 20))  # 处理一下图片像素统一，不统一的话会出现维度问题
        # img = tf.resize(img, (122, 20), mode='reflect')
        img_list.append(img)
        label_list.append(label_text)

    img_np = np.array(img_list)

    label_np = np.array(label_list, dtype=np.string_)
    print('数据集标签顺序：\n', label_np)  # 打印label信息

    # 'a' ，如果已经有这个名字的h5文件存在将不会打开，目的为了防止误删信息。
    # ‘w' ，如果有同名文件也能打开，但会覆盖上次的内容。
    with h5py.File('datasets/captcha.h5', 'w') as f:
        f.create_dataset('training_captcha_img', data=img_np)  # 创建两个数据集，分别为training_cat
        f.create_dataset('training_captcha_label', data=label_np)  # 和training_label的数组集

        f.close()


def convert_img(img_path_dir=r"D:\aitest\captcha_img_test_new"):
    # 处理全部图片为RGB通道
    for dir_image in os.listdir(img_path_dir):  # 遍历二级文件每个图片
        label_text = dir_image.split('.')[0]
        img = cv2.imread(os.path.join(img_path_dir, dir_image))
        print(img.shape)
        # RGBA的图片shape[2]为4
        if img.shape[2] >= 4:
            print(label_text)
            img = Image.open(r"{}\{}.jpg".format(img_path_dir, label_text)).convert("RGB")
            cv2.imwrite(r"{}\{}.jpg".format(img_path_dir, label_text), img)


# run
save_image_to_h5py('D:\\aitest\\captcha_img_test_new\\')

# convert_img()