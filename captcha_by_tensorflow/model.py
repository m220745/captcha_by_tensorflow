# -*- coding: utf-8 -*-
# @Time : 2021/1/22 14:36
# @Author : xiaojie
# @File : model.py
# @Software: PyCharm

import numpy as np
import cv2
from config import *


class verification_code():

    def __init__(self, model_type="pd", model_pd_path=model_pd_path, checkpoint_meta_path=checkpoint_meta_path,
                 checkpoint_dir=checkpoint_dir):
        """
        初始化识别验证码模型

        :param model_type: 加载的模型类型 pd 或者 ckpt (首选pd)
        :param model_pd_path: 如果model_type为pd时pd模型文件的路径
        :param checkpoint_meta_path: 如果model_type为ckpt时ckpt模型的.meta文件的路径
        :param checkpoint_dir: 如果model_type为ckpt时ckpt模型所在的父级目录路径
        """
        if model_type == "pd":
            with tf.gfile.FastGFile(model_pd_path, 'rb') as f:
                self.graph = tf.GraphDef()  # 生成图
                self.graph.ParseFromString(f.read())  # 图加载模型
                self.sess = tf.Session()
                tf.import_graph_def(self.graph, name='')
                # self.sess.graph.as_default()  # 可有可无
        else:
            self.graph = tf.Graph()
            self.saver = tf.train.import_meta_graph(checkpoint_meta_path, graph=self.graph)
            self.sess = tf.Session(graph=self.graph)
            self.ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            self.saver.restore(self.sess, self.ckpt)

        self.output = self.sess.graph.get_operation_by_name('op_to_store').outputs[0]
        self.X = self.sess.graph.get_operation_by_name('X').outputs[0]
        self.keep_prob = self.sess.graph.get_operation_by_name('keep_prob').outputs[0]

    def detect(self, file):
        """
        识别验证码
        :param file: 可以是文件路径或者是cv2读取的ndarray对象
        :return:
        """
        if type(file) is str:
            image = cv2.imread(file)
        else:
            image = file
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        output_rate = tf.reshape(self.output, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
        predict = tf.argmax(output_rate, 2)
        text_list, rate_list = self.sess.run([predict, output_rate], feed_dict={self.X: [image], self.keep_prob: 1})
        tmptext = text_list[0].tolist()
        text = ''
        for i in range(len(tmptext)):
            text = text + CHAR_SET[tmptext[i]]
        print('识别结果：', text)
        return text


if __name__ == '__main__':
    model = verification_code()
    model.detect(r"E:\PycharmProjects\tensorflow_test\selenium_demo\tmp\error_img\59RJ6B.jpg")
