# -*- coding: utf-8 -*-
import os
from os import listdir
from os.path import join
import tensorflow as tf
import numpy as np
import scipy.io as sio
from tf_utils_5 import random_mini_batches, convert_to_one_hot
from fusion import fusion_m
from tensorflow.python.framework import ops
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def create_placeholders(n_x1, n_x2, n_x3, n_x4, n_x5, n_y):
    isTraining = tf.placeholder_with_default(True, shape=())
    x1 = tf.placeholder(tf.float32, [None, n_x1], name="x1")
    x2 = tf.placeholder(tf.float32, [None, n_x2], name="x2")
    x3 = tf.placeholder(tf.float32, [None, n_x3], name="x3")
    x4 = tf.placeholder(tf.float32, [None, n_x4], name="x4")
    x5 = tf.placeholder(tf.float32, [None, n_x5], name="x5")
    x1_full = tf.placeholder(tf.float32, [None, n_x1], name="x1_full")
    x2_full = tf.placeholder(tf.float32, [None, n_x2], name="x2_full")
    x3_full = tf.placeholder(tf.float32, [None, n_x3], name="x3_full")
    x4_full = tf.placeholder(tf.float32, [None, n_x4], name="x4_full")
    x5_full = tf.placeholder(tf.float32, [None, n_x5], name="x5_full")
    y = tf.placeholder(tf.float32, [None, n_y], name="Y")

    return x1, x2, x3, x4, x5, x1_full, x2_full, x3_full, x4_full, x5_full, y, isTraining


def initialize_parameters():
    tf.set_random_seed(42)

    initializer = tf.keras.initializers.VarianceScaling(seed=1)
    zero_initializer = tf.keras.initializers.Zeros()

    x1_conv_w1 = tf.Variable(initializer(shape=[1, 1, 7, 32]), name="x1_conv_w1")
    x1_conv_b1 = tf.Variable(zero_initializer(shape=[32]), name="x1_conv_b1")
    x2_conv_w1 = tf.Variable(initializer(shape=[1, 1, 3, 32]), name="x2_conv_w1")
    x2_conv_b1 = tf.Variable(zero_initializer(shape=[32]), name="x2_conv_b1")
    x3_conv_w1 = tf.Variable(initializer(shape=[1, 1, 1, 32]), name="x3_conv_w1")
    x3_conv_b1 = tf.Variable(zero_initializer(shape=[32]), name="x3_conv_b1")
    x4_conv_w1 = tf.Variable(initializer(shape=[1, 1, 12, 32]), name="x4_conv_w1")
    x4_conv_b1 = tf.Variable(zero_initializer(shape=[32]), name="x4_conv_b1")
    x5_conv_w1 = tf.Variable(initializer(shape=[1, 1, 1, 32]), name="x5_conv_w1")
    x5_conv_b1 = tf.Variable(zero_initializer(shape=[32]), name="x5_conv_b1")

    x1_conv_w2 = tf.Variable(initializer(shape=[1, 1, 32, 64]), name="x1_conv_w2")
    x1_conv_b2 = tf.Variable(zero_initializer(shape=[64]), name="x1_conv_b2")
    x2_conv_w2 = tf.Variable(initializer(shape=[1, 1, 32, 64]), name="x2_conv_w2")
    x2_conv_b2 = tf.Variable(zero_initializer(shape=[64]), name="x2_conv_b2")
    x3_conv_w2 = tf.Variable(initializer(shape=[1, 1, 32, 64]), name="x3_conv_w2")
    x3_conv_b2 = tf.Variable(zero_initializer(shape=[64]), name="x3_conv_b2")
    x4_conv_w2 = tf.Variable(initializer(shape=[1, 1, 32, 64]), name="x4_conv_w2")
    x4_conv_b2 = tf.Variable(zero_initializer(shape=[64]), name="x4_conv_b2")
    x5_conv_w2 = tf.Variable(initializer(shape=[1, 1, 32, 64]), name="x5_conv_w2")
    x5_conv_b2 = tf.Variable(zero_initializer(shape=[64]), name="x5_conv_b2")

    x1_conv_w3 = tf.Variable(initializer(shape=[3, 3, 64, 128]), name="x1_conv_w3")
    x1_conv_b3 = tf.Variable(zero_initializer(shape=[128]), name="x1_conv_b3")
    x2_conv_w3 = tf.Variable(initializer(shape=[3, 3, 64, 128]), name="x2_conv_w3")
    x2_conv_b3 = tf.Variable(zero_initializer(shape=[128]), name="x2_conv_b3")
    x3_conv_w3 = tf.Variable(initializer(shape=[3, 3, 64, 128]), name="x3_conv_w3")
    x3_conv_b3 = tf.Variable(zero_initializer(shape=[128]), name="x3_conv_b3")
    x4_conv_w3 = tf.Variable(initializer(shape=[3, 3, 64, 128]), name="x4_conv_w3")
    x4_conv_b3 = tf.Variable(zero_initializer(shape=[128]), name="x4_conv_b3")
    x5_conv_w3 = tf.Variable(initializer(shape=[3, 3, 64, 128]), name="x5_conv_w3")
    x5_conv_b3 = tf.Variable(zero_initializer(shape=[128]), name="x5_conv_b3")

    x1_conv_w4 = tf.Variable(initializer(shape=[1, 1, 128, 256]), name="x1_conv_w4")
    x1_conv_b4 = tf.Variable(zero_initializer(shape=[256]), name="x1_conv_b4")
    x2_conv_w4 = tf.Variable(initializer(shape=[1, 1, 128, 256]), name="x2_conv_w4")
    x2_conv_b4 = tf.Variable(zero_initializer(shape=[256]), name="x2_conv_b4")
    x3_conv_w4 = tf.Variable(initializer(shape=[1, 1, 128, 256]), name="x3_conv_w4")
    x3_conv_b4 = tf.Variable(zero_initializer(shape=[256]), name="x3_conv_b4")
    x4_conv_w4 = tf.Variable(initializer(shape=[1, 1, 128, 256]), name="x4_conv_w4")
    x4_conv_b4 = tf.Variable(zero_initializer(shape=[256]), name="x4_conv_b4")
    x5_conv_w4 = tf.Variable(initializer(shape=[1, 1, 128, 256]), name="x5_conv_w4")
    x5_conv_b4 = tf.Variable(zero_initializer(shape=[256]), name="x5_conv_b4")

    x1_conv_w5 = tf.Variable(initializer(shape=[1, 1, 256, 128]), name="x1_conv_w5")
    x1_conv_b5 = tf.Variable(zero_initializer(shape=[128]), name="x1_conv_b5")

    x1_conv_w6 = tf.Variable(initializer(shape=[1, 1, 128, 64]), name="x1_conv_w6")
    x1_conv_b6 = tf.Variable(zero_initializer(shape=[64]), name="x1_conv_b6")

    x1_conv_w7 = tf.Variable(initializer(shape=[1, 1, 64, 17]), name="x1_conv_w7")
    x1_conv_b7 = tf.Variable(zero_initializer(shape=[17]), name="x1_conv_b7")

    parameters = {"x1_conv_w1": x1_conv_w1, "x1_conv_b1": x1_conv_b1,
                  "x2_conv_w1": x2_conv_w1, "x2_conv_b1": x2_conv_b1,
                  "x3_conv_w1": x3_conv_w1, "x3_conv_b1": x3_conv_b1,
                  "x4_conv_w1": x4_conv_w1, "x4_conv_b1": x4_conv_b1,
                  "x5_conv_w1": x5_conv_w1, "x5_conv_b1": x5_conv_b1,
                  "x1_conv_w2": x1_conv_w2, "x1_conv_b2": x1_conv_b2,
                  "x2_conv_w2": x2_conv_w2, "x2_conv_b2": x2_conv_b2,
                  "x3_conv_w2": x3_conv_w2, "x3_conv_b2": x3_conv_b2,
                  "x4_conv_w2": x4_conv_w2, "x4_conv_b2": x4_conv_b2,
                  "x5_conv_w2": x5_conv_w2, "x5_conv_b2": x5_conv_b2,
                  "x1_conv_w3": x1_conv_w3, "x1_conv_b3": x1_conv_b3,
                  "x2_conv_w3": x2_conv_w3, "x2_conv_b3": x2_conv_b3,
                  "x3_conv_w3": x3_conv_w3, "x3_conv_b3": x3_conv_b3,
                  "x4_conv_w3": x4_conv_w3, "x4_conv_b3": x4_conv_b3,
                  "x5_conv_w3": x5_conv_w3, "x5_conv_b3": x5_conv_b3,
                  "x1_conv_w4": x1_conv_w4, "x1_conv_b4": x1_conv_b4,
                  "x2_conv_w4": x2_conv_w4, "x2_conv_b4": x2_conv_b4,
                  "x3_conv_w4": x3_conv_w4, "x3_conv_b4": x3_conv_b4,
                  "x4_conv_w4": x4_conv_w4, "x4_conv_b4": x4_conv_b4,
                  "x5_conv_w4": x5_conv_w4, "x5_conv_b4": x5_conv_b4,
                  "x1_conv_w5": x1_conv_w5, "x1_conv_b5": x1_conv_b5,
                  "x1_conv_w6": x1_conv_w6, "x1_conv_b6": x1_conv_b6,
                  "x1_conv_w7": x1_conv_w7, "x1_conv_b7": x1_conv_b7,
                  }

    return parameters


def ufz_cnn(x1, x2, x3, x4, x5, parameters, isTraining):
    x1 = tf.reshape(x1, [-1, 16, 16, 7], name="x1")
    x2 = tf.reshape(x2, [-1, 16, 16, 3], name="x2")
    x3 = tf.reshape(x3, [-1, 16, 16, 1], name="x3")
    x4 = tf.reshape(x4, [-1, 16, 16, 12], name="x4")
    x5 = tf.reshape(x5, [-1, 16, 16, 1], name="x5")

    with tf.name_scope("encoder_layer_1"):
        x1_conv_layer_z1 = tf.nn.conv2d(x1, parameters['x1_conv_w1'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b1']
        x1_conv_layer_a1 = tf.nn.relu(x1_conv_layer_z1)

        x2_conv_layer_z1 = tf.nn.conv2d(x2, parameters['x2_conv_w1'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x2_conv_b1']
        x2_conv_layer_a1 = tf.nn.relu(x2_conv_layer_z1)

        x3_conv_layer_z1 = tf.nn.conv2d(x3, parameters['x3_conv_w1'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x3_conv_b1']
        x3_conv_layer_a1 = tf.nn.relu(x3_conv_layer_z1)

        x4_conv_layer_z1 = tf.nn.conv2d(x4, parameters['x4_conv_w1'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x4_conv_b1']
        x4_conv_layer_a1 = tf.nn.relu(x4_conv_layer_z1)

        x5_conv_layer_z1 = tf.nn.conv2d(x5, parameters['x5_conv_w1'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x5_conv_b1']
        x5_conv_layer_a1 = tf.nn.relu(x5_conv_layer_z1)

    with tf.name_scope("encoder_layer_2"):
        x1_conv_layer_z2 = tf.nn.conv2d(x1_conv_layer_a1, parameters['x1_conv_w2'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b2']
        x1_conv_layer_z2_po = tf.layers.max_pooling2d(x1_conv_layer_z2, 2, 2, padding='SAME')
        x1_conv_layer_a2 = tf.nn.relu(x1_conv_layer_z2_po)

        x2_conv_layer_z2 = tf.nn.conv2d(x2_conv_layer_a1, parameters['x2_conv_w2'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x2_conv_b2']
        x2_conv_layer_z2_po = tf.layers.max_pooling2d(x2_conv_layer_z2, 2, 2, padding='SAME')
        x2_conv_layer_a2 = tf.nn.relu(x2_conv_layer_z2_po)

        x3_conv_layer_z2 = tf.nn.conv2d(x3_conv_layer_a1, parameters['x3_conv_w2'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x3_conv_b2']
        x3_conv_layer_z2_po = tf.layers.max_pooling2d(x3_conv_layer_z2, 2, 2, padding='SAME')
        x3_conv_layer_a2 = tf.nn.relu(x3_conv_layer_z2_po)

        x4_conv_layer_z2 = tf.nn.conv2d(x4_conv_layer_a1, parameters['x4_conv_w2'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x4_conv_b2']
        x4_conv_layer_z2_po = tf.layers.max_pooling2d(x4_conv_layer_z2, 2, 2, padding='SAME')
        x4_conv_layer_a2 = tf.nn.relu(x4_conv_layer_z2_po)

        x5_conv_layer_z2 = tf.nn.conv2d(x5_conv_layer_a1, parameters['x5_conv_w2'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x5_conv_b2']
        x5_conv_layer_z2_po = tf.layers.max_pooling2d(x5_conv_layer_z2, 2, 2, padding='SAME')
        x5_conv_layer_a2 = tf.nn.relu(x5_conv_layer_z2_po)

    with tf.name_scope("encoder_layer_3"):
        x1_conv_layer_z3 = tf.nn.conv2d(x1_conv_layer_a2, parameters['x1_conv_w3'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b3']
        x1_conv_layer_a3 = tf.nn.relu(x1_conv_layer_z3)

        x2_conv_layer_z3 = tf.nn.conv2d(x2_conv_layer_a2, parameters['x2_conv_w3'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x2_conv_b3']
        x2_conv_layer_a3 = tf.nn.relu(x2_conv_layer_z3)

        x3_conv_layer_z3 = tf.nn.conv2d(x3_conv_layer_a2, parameters['x3_conv_w3'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x3_conv_b3']
        x3_conv_layer_a3 = tf.nn.relu(x3_conv_layer_z3)

        x4_conv_layer_z3 = tf.nn.conv2d(x4_conv_layer_a2, parameters['x4_conv_w3'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x4_conv_b3']
        x4_conv_layer_a3 = tf.nn.relu(x4_conv_layer_z3)

        x5_conv_layer_z3 = tf.nn.conv2d(x5_conv_layer_a2, parameters['x5_conv_w3'], strides=[1, 1, 1, 1], padding='SAME') + \
                           parameters['x5_conv_b3']
        x5_conv_layer_a3 = tf.nn.relu(x5_conv_layer_z3)

    with tf.name_scope("encoder_layer_4"):
        x1_conv_layer_z4 = tf.nn.conv2d(x1_conv_layer_a3, parameters['x1_conv_w4'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b4']
        x1_conv_layer_z4_po = tf.layers.max_pooling2d(x1_conv_layer_z4, 2, 2, padding='SAME')
        x1_conv_layer_a4 = tf.nn.relu(x1_conv_layer_z4_po)

        x2_conv_layer_z4 = tf.nn.conv2d(x2_conv_layer_a3, parameters['x2_conv_w4'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x2_conv_b4']
        x2_conv_layer_z4_po = tf.layers.max_pooling2d(x2_conv_layer_z4, 2, 2, padding='SAME')
        x2_conv_layer_a4 = tf.nn.relu(x2_conv_layer_z4_po)

        x3_conv_layer_z4 = tf.nn.conv2d(x3_conv_layer_a3, parameters['x3_conv_w4'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x3_conv_b4']
        x3_conv_layer_z4_po = tf.layers.max_pooling2d(x3_conv_layer_z4, 2, 2, padding='SAME')
        x3_conv_layer_a4 = tf.nn.relu(x3_conv_layer_z4_po)

        x4_conv_layer_z4 = tf.nn.conv2d(x4_conv_layer_a3, parameters['x4_conv_w4'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x4_conv_b4']
        x4_conv_layer_z4_po = tf.layers.max_pooling2d(x4_conv_layer_z4, 2, 2, padding='SAME')
        x4_conv_layer_a4 = tf.nn.relu(x4_conv_layer_z4_po)

        x5_conv_layer_z4 = tf.nn.conv2d(x5_conv_layer_a3, parameters['x5_conv_w4'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x5_conv_b4']
        x5_conv_layer_z4_po = tf.layers.max_pooling2d(x5_conv_layer_z4, 2, 2, padding='SAME')
        x5_conv_layer_a4 = tf.nn.relu(x5_conv_layer_z4_po)

    with tf.name_scope("fusion_module"):
        fused_layer = fusion_m([x1_conv_layer_a4, x2_conv_layer_a4, x3_conv_layer_a4, x4_conv_layer_a4, x5_conv_layer_a4])
        fused_layer = tf.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same', activation='relu',
                                         kernel_initializer='he_normal', use_bias=False)(fused_layer)

    with tf.name_scope("encoder_layer_5"):
        x1_conv_layer_z5 = tf.nn.conv2d(fused_layer, parameters['x1_conv_w5'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b5']
        x1_conv_layer_a5 = tf.nn.relu(x1_conv_layer_z5)

        x1_conv_layer_z6 = tf.nn.conv2d(x1_conv_layer_a5, parameters['x1_conv_w6'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b6']
        x1_conv_layer_z6_po = tf.layers.average_pooling2d(x1_conv_layer_z6, 2, 2, padding='SAME')  # padding='SAME'
        x1_conv_layer_a6 = tf.nn.relu(x1_conv_layer_z6_po)

        x1_conv_layer_z7 = tf.nn.conv2d(x1_conv_layer_a6, parameters['x1_conv_w7'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b7']
        x1_conv_layer_z7_shape = x1_conv_layer_z7.get_shape().as_list()
        x1_conv_layer_z7_2d = tf.reshape(x1_conv_layer_z7, [-1, x1_conv_layer_z7_shape[1] * x1_conv_layer_z7_shape[2] * x1_conv_layer_z7_shape[3]])

    with tf.name_scope("l2_loss"):
        l2_loss = tf.nn.l2_loss(parameters['x1_conv_w1']) + tf.nn.l2_loss(parameters['x1_conv_w2']) + tf.nn.l2_loss(
            parameters['x1_conv_w3']) + tf.nn.l2_loss(parameters['x1_conv_w4']) + tf.nn.l2_loss(parameters['x2_conv_w1']) \
                  + tf.nn.l2_loss(parameters['x2_conv_w2']) + tf.nn.l2_loss(parameters['x2_conv_w3']) + tf.nn.l2_loss(
            parameters['x2_conv_w4']) + tf.nn.l2_loss(parameters['x3_conv_w1']) + tf.nn.l2_loss(parameters['x3_conv_w2']) + tf.nn.l2_loss(
            parameters['x3_conv_w3']) + tf.nn.l2_loss(parameters['x3_conv_w4']) + tf.nn.l2_loss(parameters['x1_conv_w5']) + tf.nn.l2_loss(
            parameters['x1_conv_w6']) + tf.nn.l2_loss(parameters['x1_conv_w7']) + tf.nn.l2_loss(parameters['x4_conv_w1']) + tf.nn.l2_loss(
            parameters['x4_conv_w2']) + tf.nn.l2_loss(parameters['x4_conv_w3']) + tf.nn.l2_loss(parameters['x4_conv_w4']) + tf.nn.l2_loss(
            parameters['x5_conv_w1']) + tf.nn.l2_loss(parameters['x5_conv_w2']) + tf.nn.l2_loss(parameters['x5_conv_w3']) + tf.nn.l2_loss(
            parameters['x5_conv_w4'])

    return x1_conv_layer_z7_2d, l2_loss


def dice_loss(y_es, y_re):
    intersection = tf.reduce_sum(y_es * y_re)
    dice_coefficient = (2.0 * intersection) / (tf.reduce_sum(y_es) + tf.reduce_sum(y_re))
    dice_loss = 1.0 - dice_coefficient
    return dice_loss

def mynetwork_optimization(y_es, y_re, l2_loss, reg, learning_rate, global_step):
    with tf.name_scope("cost"):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_es, labels=y_re))
        dice = dice_loss(tf.nn.softmax(y_es), y_re)  # Adding DICE loss here
        total_loss = cross_entropy_loss + reg * l2_loss + dice  # Combining cross-entropy, L2, and DICE loss

    with tf.name_scope("optimization"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, global_step=global_step)

    return total_loss, optimizer


def train(x1_train_set, x2_train_set, x3_train_set, x4_train_set, x5_train_set, x1_train_set_full, x2_train_set_full, x3_train_set_full,
                    x4_train_set_full, x5_train_set_full, x1_test_set, x2_test_set, x3_test_set, x4_test_set, x5_test_set, y_train_set, y_test_set,
                    learning_rate_base=0.001, beta_reg=0.001, num_epochs=100, minibatch_size=64, print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 1
    (m, n_x1) = x1_train_set.shape
    (m, n_x2) = x2_train_set.shape
    (m, n_x3) = x3_train_set.shape
    (m, n_x4) = x4_train_set.shape
    (m, n_x5) = x5_train_set.shape
    (m, n_y) = y_train_set.shape

    costs = []
    costs_dev = []
    train_acc = []
    val_acc = []
    correct_prediction = 0

    # Create Placeholders of shape (n_x, n_y)
    x1, x2, x3, x4, x5, x1_full, x2_full, x3_full, x4_full, x5_full, y, isTraining = create_placeholders(n_x1, n_x2, n_x3, n_x4, n_x5, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    with tf.name_scope("network"):
        joint_layer, l2_loss = ufz_cnn(x1, x2, x3, x4, x5, parameters, isTraining)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 30 * m / minibatch_size, 0.5, staircase=True)

    with tf.name_scope("optimization"):
        cost, optimizer = mynetwork_optimization(joint_layer, y, l2_loss, beta_reg, learning_rate, global_step)

    with tf.name_scope("metrics"):
        joint_layerT = tf.transpose(joint_layer)
        yT = tf.transpose(y)
        correct_prediction = tf.equal(tf.argmax(joint_layerT), tf.argmax(yT))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Initialize all the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs + 1):

            epoch_cost = 0.  # Defines a cost related to an epoch
            epoch_acc = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(x1_train_set, x2_train_set, x3_train_set, x4_train_set, x5_train_set,
                                              x1_train_set_full, x2_train_set_full, x3_train_set_full, x4_train_set_full, x5_train_set_full,
                                              y_train_set, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x1_full, batch_x2_full, batch_x3_full, batch_x4_full, batch_x5_full, batch_y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost, minibatch_acc = sess.run(
                    [optimizer, cost, accuracy],
                    feed_dict={x1: batch_x1, x2: batch_x2, x3: batch_x3, x4: batch_x4, x5: batch_x5,
                               x1_full: batch_x1_full, x2_full: batch_x2_full, x3_full: batch_x3_full, x4_full: batch_x4_full, x5_full: batch_x5_full,
                               y: batch_y, isTraining: True})

                epoch_cost += minibatch_cost / (num_minibatches + 1)
                epoch_acc += minibatch_acc / (num_minibatches + 1)

            trainfeature, epoch_cost_dev_tr, epoch_acc_dev_tr = sess.run(
                [joint_layerT, cost, accuracy],
                feed_dict={x1: x1_train_set, x2: x2_train_set, x3: x3_train_set, x4: x4_train_set, x5: x5_train_set,
                           x1_full: x1_train_set, x2_full: x2_train_set, x3_full: x3_train_set, x4_full: x4_train_set, x5_full: x5_train_set,
                           y: y_train_set, isTraining: False})
            testfeature, epoch_cost_dev_te, epoch_acc_dev_te = sess.run(
                [joint_layerT, cost, accuracy],
                feed_dict={x1: x1_test_set, x2: x2_test_set, x3: x3_test_set, x4: x4_test_set, x5: x5_test_set,
                           x1_full: x1_test_set, x2_full: x2_test_set, x3_full: x3_test_set, x4_full: x4_test_set, x5_full: x5_test_set,
                           y: y_test_set, isTraining: False})

            # Print the cost every 10 epoch
            if print_cost == True and epoch % 10 == 0:
                print("epoch %i: mini_loss: %f, mini_acc: %f, Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (
                    epoch, epoch_cost, epoch_acc, epoch_cost_dev_tr, epoch_cost_dev_te, epoch_acc_dev_tr, epoch_acc_dev_te))

            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost_dev_tr)
                train_acc.append(epoch_acc_dev_tr)
                costs_dev.append(epoch_cost_dev_te)
                val_acc.append(epoch_acc_dev_te)

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        print("save model")
        save_path = saver.save(sess, "./model/model.ckpt")
        print("save model:{0} Finished".format(save_path))

        return parameters, val_acc, testfeature, trainfeature


random_seed = 42
np.random.seed(random_seed)

dataset = DatasetFromFolder(r"G:\file\dataset512")
MIITR, MIITE, GIUTR, GIUTE, BDHTR, BDHTE, POITR, POITE, DSMTR, DSMTE, LabTR, LabTE = dataset
Y_train = convert_to_one_hot(LabTR.astype("int64") - 1, 17)
Y_test = convert_to_one_hot(LabTE.astype("int64") - 1, 17)
Y_train = Y_train.T
Y_test = Y_test.T

parameters, val_acc, testfeature, trainfeature = train(MIITR, GIUTR, BDHTR, POITR, DSMTR, MIITR, GIUTR, BDHTR, POITR, DSMTR,
                                                                 MIITE, GIUTE, BDHTE, POITE, DSMTE, Y_train, Y_test)

max_val_acc = max(val_acc)
max_val_acc_index = np.argmax(val_acc)
# Save features
sio.savemat(rf'G:\file\CNN\result\testfeature.mat', {'testfeature': testfeature})
sio.savemat(rf'G:\file\CNN\result\trainfeature.mat', {'trainfeature': trainfeature})
print(f"Training completed, Max Validation Accuracy: {max_val_acc} (Epoch: {max_val_acc_index})")
