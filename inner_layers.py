################################################################################
# Michael Guerzhoy and Davi Frossard, 2016
# AlexNet implementation in TensorFlow, with weights
# Details:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
# With code from https://github.com/ethereon/caffe-tensorflow
# Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
import numpy as np
import time
from scipy.misc import imread, imsave
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from caffe_classes import class_names

train_x = zeros((1, 227, 227, 3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]
################################################################################

net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


y_ = tf.placeholder(tf.float32, [None, 1000])
x_ = tf.Variable(np.zeros((1, 227, 227, 3)).astype(np.float32))
# x_ = tf.Variable(np.random.random((1, 227, 227, 3)).astype(np.float32))
# conv1
# conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11
k_w = 11
c_o = 96
s_h = 4
s_w = 4
conv1W = tf.Variable(net_data["conv1"][0], trainable=False)
conv1b = tf.Variable(net_data["conv1"][1], trainable=False)
conv1_in = conv(x_, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

# lrn1
radius = 2
alpha = 2e-05
beta = 0.75
bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                          depth_radius=radius,
                                          alpha=alpha,
                                          beta=beta,
                                          bias=bias)

# maxpool1
k_h = 3
k_w = 3
s_h = 2
s_w = 2
padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

# conv2
k_h = 5
k_w = 5
c_o = 256
s_h = 1
s_w = 1
group = 2
conv2W = tf.Variable(net_data["conv2"][0], trainable=False)
conv2b = tf.Variable(net_data["conv2"][1], trainable=False)
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)

# lrn2
radius = 2
alpha = 2e-05
beta = 0.75
bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                          depth_radius=radius,
                                          alpha=alpha,
                                          beta=beta,
                                          bias=bias)

# maxpool2
k_h = 3
k_w = 3
s_h = 2
s_w = 2
padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

# conv3
k_h = 3
k_w = 3
c_o = 384
s_h = 1
s_w = 1
group = 1
conv3W = tf.Variable(net_data["conv3"][0], trainable=False)
conv3b = tf.Variable(net_data["conv3"][1], trainable=False)
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

# conv4
k_h = 3
k_w = 3
c_o = 384
s_h = 1
s_w = 1
group = 2
conv4W = tf.Variable(net_data["conv4"][0], trainable=False)
conv4b = tf.Variable(net_data["conv4"][1], trainable=False)
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

# conv5
k_h = 3
k_w = 3
c_o = 256
s_h = 1
s_w = 1
group = 2
conv5W = tf.Variable(net_data["conv5"][0], trainable=False)
conv5b = tf.Variable(net_data["conv5"][1], trainable=False)
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

# maxpool5
k_h = 3
k_w = 3
s_h = 2
s_w = 2
padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

# fc6
fc6W = tf.Variable(net_data["fc6"][0], trainable=False)
fc6b = tf.Variable(net_data["fc6"][1], trainable=False)
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

# fc7
fc7W = tf.Variable(net_data["fc7"][0], trainable=False)
fc7b = tf.Variable(net_data["fc7"][1], trainable=False)
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

# fc8
fc8W = tf.Variable(net_data["fc8"][0], trainable=False)
fc8b = tf.Variable(net_data["fc8"][1], trainable=False)
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# prob
prob = tf.nn.softmax(fc8)


#******************************* Q1 *************************************************
def visualizationOfNeuronsViaInput(lamda, neuron, input_x, num_iter, neuron_max_val):
    l2_reg = tf.reduce_mean(tf.square(input_x))
    la = np.float32(lamda)
    target = -(neuron - la * l2_reg)
    train_step = tf.train.AdamOptimizer(1e-2).minimize(target)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_iter):
            g1, g2, g3 = sess.run([train_step, target, neuron])
            print("neuron value: {0:0.3f} | step:{1}".format(g3, i))
            if g3 > neuron_max_val:
                break

        vis_input = sess.run(input_x)
        vis_input = vis_input[0, :, :, :]
        plt.imshow((vis_input[:, :, :]).astype(np.uint8))
        plt.show()
        imsave('input_im.png', vis_input)


#******************************* Q2 *************************************************
def visualizationOfNeuronsUsingFFT(lamda, neuron, input_x, num_iter, neuron_max_val):
    fft = tf.fft(tf.cast(x_, tf.complex64))
    # W = tf.cast(tf.norm(fft), tf.float32)
    f_reg = tf.log(tf.maximum(tf.abs(fft), 1E-02))
    # f_reg = tf.reduce_mean((tf.abs(fft) - 1 / W))
    la = np.float32(lamda)
    target = -(neuron - la * f_reg)
    train_step = tf.train.AdamOptimizer(1e-2).minimize(target)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_iter):
            # tf.clip_by_norm(input_x, 2)

            g1, g2, g3 = sess.run([train_step, target, neuron])
            print("neuron value: {0:0.3f} | step:{1}".format(g3, i))
            if g3 > neuron_max_val:
                break

        vis_input = sess.run(input_x)
        vis_input = vis_input[0, :, :, :]
        plt.imshow((vis_input[:, :, :]).astype(np.uint8))
        plt.show()
        imsave('input_im_fft.png', vis_input)


if __name__ == '__main__':
    # visualizationOfNeuronsViaInput(0.05, fc8[0][808], x_, 300, 100)
    # visualizationOfNeuronsViaInput(0.05, conv3_in[0][4][4][4], x_, 500, 50)
    # visualizationOfNeuronsUsingFFT(0.7, fc8[0][99], x_, 500, 100)
    visualizationOfNeuronsUsingFFT(0.5, fc8[0][808], x_, 500, 100)






