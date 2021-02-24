# -*- coding: utf-8 -*-
# @Time : 2021/1/22 9:46
# @Author : xiaojie
# @File : cnn_model.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import cnn_utils as util
from config import *
# 指定GPU运行
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class verification_code():
    def_sess = None
    def_output = None
    def_saver = None
    def_module_file = None

    # 加载数据
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = util.load_dataset()
    # 查看一下储存的验证码图片和对应的标签值
    # index = 5
    # print(Y_train_orig[:, index])
    # plt.imshow(X_train_orig[index])
    # plt.show()
    # print("y = " + str(np.squeeze(Y_train_orig[:, index])))

    # 简单的归一化处理
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # 生成数据集中标签对应的 one hot编码
    Y_train = util.convert_to_one_hot(Y_train_orig, CHAR_SET_LEN, CAPTCHA_LEN)
    Y_test = util.convert_to_one_hot(Y_test_orig, CHAR_SET_LEN, CAPTCHA_LEN)

    # 检查一下各个变量的维度
    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    # TODO
    X, Y = util.create_placeholders(IMAGE_HEIGHT, IMAGE_WIDTH, 3, CHAR_SET_LEN * CAPTCHA_LEN)
    print("X = " + str(X))
    print("Y = " + str(Y))

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # 使用dropout正则化，防止过拟合

    # 构建CNN网络
    def captcha_cnn_network(self, w_alpha=0.01, b_alpha=0.1):

        # TODO
        # 把 X reshape 成 IMAGE_HEIGHT*IMAGE_WIDTH*1的格式,输入的是灰度图片，所以通道数是1;
        # shape 里的-1表示数量不定，根据实际情况获取，这里为每轮迭代输入的图像数量（batchsize）的大小;
        x = tf.reshape(self.X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

        # TODO
        # 搭建第一层卷积层
        # shape[3, 3, 1, 32]里前两个参数表示卷积核尺寸大小，即patch;
        # 第三个参数是图像通道数，第四个参数是该层卷积核的数量，有多少个卷积核就会输出多少个卷积特征图像
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 3, 32]))
        # 每个卷积核都配置一个偏置量，该层有多少个输出，就应该配置多少个偏置量
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))

        # 图片和卷积核卷积，并加上偏执量，卷积结果20x122x32
        # tf.nn.conv2d() 函数实现卷积操作
        # tf.nn.conv2d()中的padding用于设置卷积操作对边缘像素的处理方式，在tf中有VALID和SAME两种模式
        # padding='SAME'会对图像边缘补0,完成图像上所有像素（特别是边缘象素）的卷积操作
        # padding='VALID'会直接丢弃掉图像边缘上不够卷积的像素
        # strides：卷积时在图像每一维的步长，是一个一维的向量，长度4，并且strides[0]=strides[3]=1
        # tf.nn.bias_add() 函数的作用是将偏置项b_c1加到卷积结果value上去;
        # 注意这里的偏置项b_c1必须是一维的，并且数量一定要与卷积结果value最后一维数量相同
        # tf.nn.relu() 函数是relu激活函数，实现输出结果的非线性转换，即features=max(features, 0)，输出tensor的形状和输入一致
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        # tf.nn.max_pool()函数实现最大池化操作，进一步提取图像的抽象特征，并且降低特征维度
        # ksize=[1, 2, 2, 1]定义最大池化操作的核尺寸为2*2, 池化结果10x10x32 卷积结果乘以池化卷积核
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # tf.nn.dropout是tf里为了防止或减轻过拟合而使用的函数，一般用在全连接层;
        # Dropout机制就是在不同的训练过程中根据一定概率（大小可以设置，一般情况下训练推荐0.5）随机扔掉（屏蔽）一部分神经元，
        # 不参与本次神经网络迭代的计算（优化）过程，权重保留但不做更新;
        # tf.nn.dropout()中 keep_prob用于设置概率，需要是一个占位变量，在执行的时候具体给定数值
        conv1 = tf.nn.dropout(conv1, self.keep_prob)
        # 原图像HEIGHT = 20 WIDTH = 122，经过神经网络第一层卷积（图像尺寸不变、特征×32）、池化（图像尺寸缩小一半，特征不变）之后;
        # 输出大小为 10*61*32

        # 搭建第二层卷积层
        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)
        # 原图像HEIGHT = 20 WIDTH = 122，经过神经网络第一层后输出大小为 10*61*32
        # 经过神经网络第二层运算后输出为 5*31*64 (10*61的图像经过2*2的卷积核池化，padding为SAME，输出维度是5*31*64)

        # 搭建第三层卷积层
        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        # 原图像HEIGHT = 20 WIDTH = 122，经过神经网络第一层后输出大小为 10*61*32 经过第二层后输出为 5*31*64
        # 经过神经网络第二层运算后输出为 5*31*64 ; 经过第三层输出为 3*16*64，这个参数很重要，决定量后边全连接层的维度

        # P = tf.contrib.layers.flatten(conv3)

        # 搭建全连接层
        # 二维张量，第一个参数3*16*64的patch，这个参数由最后一层卷积层的输出决定，第二个参数代表卷积个数共1024个，即输出为1024个特征
        w_d = tf.Variable(w_alpha * tf.random_normal([3 * 16 * 64, 1024]))
        # 偏置项为1维，个数跟卷积核个数保持一致
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        # w_d.get_shape()作用是把张量w_d的形状转换为元组tuple的形式，w_d.get_shape().as_list()是把w_d转为元组再转为list形式
        # w_d 的 形状是[ 3 * 16 * 64, 1024]，w_d.get_shape().as_list()结果为 3*16*64=3072 ;
        # 所以tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])的作用是把最后一层隐藏层的输出转换成一维的形式
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        # tf.matmul(dense, w_d)函数是矩阵相乘，输出维度是 -1*1024
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self.keep_prob)
        # 经过全连接层之后，输出为 一维，1024个向量

        # w_out定义成一个形状为 [1024, 36 * 6] = [1024, 216]
        w_out = tf.Variable(w_alpha * tf.random_normal([1024, CHAR_SET_LEN * CAPTCHA_LEN]))
        b_out = tf.Variable(b_alpha * tf.random_normal([CHAR_SET_LEN * CAPTCHA_LEN]))
        # out 的输出为 6*36 的向量， 6代表识别结果的验证码长度，36是每一位上可能的结果（0到大写英文字母Z）
        out = tf.add(tf.matmul(dense, w_out), b_out, name="op_to_store")
        # out = tf.nn.softmax(out)
        # 输出神经网络在当前参数下的预测值
        return out

    # 模型训练
    def train_captcha_cnn_network(self, step_cnt=2000, minibatch_size=16, learning_rate=0.0001, keep_prob=0.5):
        """
        训练模型
        :param step_cnt: 迭代轮数
        :param minibatch_size: 批量获取样本数量
        :param learning_rate: 学习率
        :param keep_prob: dropout 保留的神经元比例
        :return:
        """
        # TODO
        (m, n_H0, n_W0, n_C0) = self.X_train.shape
        seed = 3
        costs = []
        accs = []
        # 加载网络结构
        output = self.captcha_cnn_network()

        # 损失函数
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.Y))
        # Adam算法优化器，学习率开始设置大一点，然后慢慢减小
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # 评估准确率
        predict = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            for step in range(step_cnt):
                minibatch_cost = 0.
                num_minibatches = int(m / minibatch_size)
                seed = seed + 1
                minibatches = util.random_mini_batches(self.X_train, self.Y_train, minibatch_size, seed)
                bach_index = 0
                for minibatch in minibatches:
                    (batch_x, batch_y) = minibatch
                    _, loss_ = sess.run([optimizer, loss],
                                        feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: keep_prob})

                    minibatch_cost += loss_ / num_minibatches
                    print('step:', step, "minibatch-", bach_index, 'minibatch_cost:', minibatch_cost)
                    bach_index += 1
                # 每10个epoch评估一次准确率
                if step % 10 == 0:

                    print("Cost after epoch %i: %f" % (step, minibatch_cost))
                    minibatches = util.random_mini_batches(self.X_test, self.Y_test, minibatch_size, seed)
                    (batch_x_test, batch_y_test) = minibatches[0]
                    acc = sess.run(accuracy, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    print('step:', step, 'acc:', acc)
                    accs.append(acc)
                    costs.append(minibatch_cost)

                    # if acc > accuracy_rate:
                    #     # 保存为.ckpt模型
                    #     saver.save(sess, checkpoint_path, global_step=step)
                    #
                    #     # 保存为.pd模型
                    #     output_rate = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name="output_rate")
                    #     predict = tf.argmax(output_rate, 2, name="predict")
                    #     constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                    #                                                                   ['op_to_store'])
                    #     with tf.gfile.FastGFile(model_pd_path, mode='wb') as f:
                    #         f.write(constant_graph.SerializeToString())
                    #
                    #     break

                step += 1

            # 跑完全部迭代才保存最终模型

            # 保存为.ckpt模型
            saver.save(sess, checkpoint_path, global_step=step)

            # 保存为.pd模型
            output_rate = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name="output_rate")
            predict = tf.argmax(output_rate, 2, name="predict")
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                          ['op_to_store'])
            with tf.gfile.FastGFile(model_pd_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())


            # 画出成本的曲线图
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

            # 画出准确率的曲线图
            plt.plot(np.squeeze(accs))
            plt.ylabel('acc')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

            # 分别计算一下在训练集和测试集上面的预测精准度
            print("Train Accuracy:",
                  sess.run(accuracy, feed_dict={self.X: self.X_train, self.Y: self.Y_train, self.keep_prob: 1.}))
            print("Test Accuracy:",
                  sess.run(accuracy, feed_dict={self.X: self.X_test, self.Y: self.Y_test, self.keep_prob: 1.}))

        pass

    # 模型预测
    def predict_captcha(self, captcha_image):
        """
        预测验证码
        :param captcha_image: 需要预测的图片路径
        :return: 预测结果
        """
        # 加载网络结构

        output = self.captcha_cnn_network()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # model_path = 'model/'
            module_file = tf.train.latest_checkpoint(checkpoint_dir)
            if module_file is not None:
                saver.restore(sess, module_file)
            output_rate = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
            predict = tf.argmax(output_rate, 2)
            text_list, rate_list = sess.run([predict, output_rate],
                                            feed_dict={self.X: [captcha_image], self.keep_prob: 1})
            tmptext = text_list[0].tolist()
            text = ''
            for i in range(len(tmptext)):
                text = text + CHAR_SET[tmptext[i]]
            print('识别结果：', text)
            return text
