# -*- coding: utf-8 -*-
# @Time : 2021/1/20 16:20
# @Author : xiaojie
# @File : cnn_utils.py
# @Software: PyCharm

import cv2
import numpy as np
import h5py
import math

from sklearn.model_selection import train_test_split
from config import *

np.random.seed(1)


def load_dataset():
    # 1 读取数据集
    train_dataset = h5py.File('datasets/captcha.h5', "r")
    # print(train_dataset.keys())
    # 2 划分数据集
    train_set_x_orig, test_set_x_orig, train_set_y_orig, test_set_y_orig = train_test_split(
        np.array(train_dataset["training_captcha_img"][:]), np.array(train_dataset["training_captcha_label"][:]),
        test_size=0.1, random_state=15)

    # TODO
    # 图片灰度化处理=============================================================
    # train_set_x_orig_templist = []
    # test_set_x_orig_templist = []
    # for i in train_set_x_orig:
    #     train_set_x_orig_templist.append(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))
    # for i in test_set_x_orig:
    #     test_set_x_orig_templist.append(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))
    # train_set_x_orig = np.array(train_set_x_orig_templist)
    # test_set_x_orig = np.array(test_set_x_orig_templist)
    # 图片灰度化处理end===========================================================

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # print("训练集的特征值是:\n", train_set_x_orig)
    # print("测试集的特征值是:\n", test_set_x_orig)
    # print("训练集的目标值是:\n", train_set_y_orig)
    # print("测试集的目标值是:\n", test_set_y_orig)
    # print("训练集的目标值形状:\n", train_set_y_orig.shape)
    # print("测试集的目标值形状:\n", test_set_y_orig.shape)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def text2label(text, C, L):
    """
    验证码文本信息one-hot编码
    :param text: 文本字符串
    :param C: 分类变量类别数量
    :param L: 分类字符长度
    :return:
    """
    label = np.zeros(L * C)
    for i in range(len(text)):
        idx = i * C + CHAR_SET.index(text[i])
        label[idx] = 1
    return label


def convert_to_one_hot(Y, C, L):
    temp_labels_list = []
    labels = Y[0]
    for i in range(0, len(labels)):
        temp_labels_list.append(text2label(str(labels[i], encoding="utf-8"), C, L))
    Y = np.array(temp_labels_list)
    return Y


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    参数:
    n_H0 -- 图像矩阵的高
    n_W0 -- 图像矩阵的宽
    n_C0 -- 图像矩阵的深度
    n_y -- 标签类别数量，因为是数字0到英文字母Z，所以数量是36

    返回值:
    X -- 样本数据的占位符
    Y -- 标签的占位符
    """

    # 下面使用None来表示样本数量，表示当前还不确定样本数量
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name="X") # 如果是3维图片，使用这一行
    # TODO
    # X = tf.placeholder(tf.float32, [None, n_H0, n_W0], name="X")  # 处理成黑白图片后，是二维数据，使用这一行
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y
