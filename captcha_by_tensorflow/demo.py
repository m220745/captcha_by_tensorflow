# -*- coding: utf-8 -*-
# @Time : 2021/1/22 10:48
# @Author : xiaojie
# @File : demo.py
# @Software: PyCharm

import time
import numpy as np
import cv2
import cnn_model
from config import *

if __name__ == '__main__':
    # 模型的类
    model = cnn_model.verification_code()

    action = 1
    if action == 0:
        # 训练模型
        start = time.time()
        try:
            model.train_captcha_cnn_network(step_cnt=2000, minibatch_size=64, learning_rate=0.0001, keep_prob=0.5)
        except Exception as e:
            print(e)
        end = time.time()
        print("Run Time: ", end - start)

    elif action == 1:
        # 识别验证码（模型预测）
        image = cv2.imread(r"D:\aitest\captcha_img_test\EK111V.jpg")
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        model.predict_captcha(image)
        pass

    pass
