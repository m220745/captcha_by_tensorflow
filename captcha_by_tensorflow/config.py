# -*- coding: utf-8 -*-
# @Time : 2021/1/21 11:06
# @Author : xiaojie
# @File : config.py
# @Software: PyCharm

import string
import os

# import tensorflow as tf
# 以下两行代码以使用tf1.x版本
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf = tf
# 验证码目录
DATA_DIR = "D:\\aitest\\captcha_image\\"

# 生成验证码的字符集
CHAR_SET = string.digits + string.ascii_uppercase  # 验证码里的所有字符

CHAR_SET_LEN = len(CHAR_SET)

# 验证码长度
CAPTCHA_LEN = 6

# 图像大小
IMAGE_HEIGHT = 20
IMAGE_WIDTH = 122
IMAGE_C = 3

# 训练时保存的文件夹
checkpoint_path = "model/crack_captcha.ctpk"
checkpoint_meta_path = "model/crack_captcha.ctpk-2000.meta"
checkpoint_dir = os.path.dirname(checkpoint_path)

model_pd_path = "model/crack_captcha.pd"
model_pd_dir = os.path.dirname(model_pd_path)

# 到多少准确度以后就停止训练
accuracy_rate = 0.95
