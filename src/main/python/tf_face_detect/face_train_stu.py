import argparse
import json
import math
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from google.protobuf import json_format

parser = argparse.ArgumentParser(description='frozen_layers to this script')
parser.add_argument('--unfrozen', type=int, default=None)
parser.add_argument('-p', '--path', help='the root path ')
args = parser.parse_args()

path = args.path
block_idx = args.unfrozen

path = "./model_info/"

block_1, block_2, block_3, block_4, block_5 = False, False, False, False, False
if block_idx == 1:
    block_1 = True
elif block_idx == 2:
    block_2 = True
elif block_idx == 3:
    block_3 = True
elif block_idx == 4:
    block_4 = True
elif block_idx == 5:
    block_5 = True
else:
    block_1, block_2, block_3, block_4, block_5 = True, True, True, True, True

nb_classes = 10575
BATCH_SIZE = 64
LambdaMin = 5.0
LambdaMax = 1500.0
load_check = False
data_path = '/Users/voyager/Downloads/facenet/data/CASIA-WebFace-aligned'
checkpoints_dir = 'checkpoints/'
block_nums = 80
generated_graph = True


def prelu(input, name, trainable=True):
    alphas = tf.get_variable(name + '_alphas', input.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32,
                             trainable=trainable)
    pos = tf.nn.relu(input)
    neg = alphas * tf.nn.relu(-input)
    return tf.add(pos, neg, name=name)


def weight_variable(shape, stddev=0.2, name=None, trainable=True):
    initial = tf.glorot_uniform_initializer()
    if name is None:
        return tf.Variable(initial, trainable=trainable)
    else:
        return tf.get_variable(name, shape=shape, initializer=initial, trainable=trainable)


def ce_loss(logit, label, reg_ratio=0.):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label))
    return cross_entropy_loss


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_x = X[permutation, :]
    shuffled_y = Y[permutation]

    num_complete_mini_batches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_mini_batches):
        mini_batch_x = shuffled_x[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_y = shuffled_y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches


def get_img_path_and_label(path, block=10):
    image_cate = os.listdir(path)
    for p in image_cate:
        if p.endswith('txt'):
            image_cate.remove(p)
    cate_num = []
    for i in range(len(image_cate)):
        cate_num.append(len(os.listdir(os.path.join(path, image_cate[i]))))
    size = sum(cate_num)
    y = np.zeros(shape=(size,), dtype=np.int32)
    img_path = [''] * size
    s = 0
    for i in range(len(image_cate)):
        cate_dir = os.listdir(os.path.join(path, image_cate[i]))
        for img in cate_dir:
            img_path[s] = os.path.join(path, image_cate[i] + '/' + img)
            y[s] = int(image_cate[i])
            s += 1
    rl1 = list(range(y.shape[0]))
    random.shuffle(rl1)
    img_path = np.array(img_path)[rl1]
    y = y[rl1]
    data_block = []
    block_size = size // block
    for j in range(block):
        data_block.append((img_path[j * block_size: min((j + 1) * block_size, size)],
                           y[j * block_size: min((j + 1) * block_size, size)]))
    data_block.append((img_path[-(size - block * block_size):], y[-(size - block * block_size):]))
    return data_block


def get_data(data_block):
    data_path, y = data_block
    size = data_path.shape[0] * 2
    x = np.zeros(shape=(size, 112, 96, 3), dtype=np.float32)
    s = 0
    for path in data_path:
        im = Image.open(str(path)).resize((96, 112))
        imflip = im.transpose(Image.FLIP_LEFT_RIGHT)
        x[s] = np.array(im)
        x[s + 1] = np.array(imflip)
        s += 2
    y = y.repeat(2)
    y = np.eye(nb_classes)[y]
    return x, y


def gen_graph(input_x, input_y, learning_rate, scope='teacher', reuse=False):

    with tf.variable_scope(scope, reuse=reuse) as sc:
        #
        conv_11 = tf.nn.conv2d(input_x,
                               weight_variable([3, 3, 3, 64], name='filter_11', trainable=block_1),
                               strides=[1, 2, 2, 1], padding='SAME', name='conv_11')
        prelu_11 = prelu(conv_11, name='prelu_11', trainable=block_1)

        conv_12 = tf.nn.conv2d(prelu_11,
                               weight_variable([3, 3, 64, 64], name='filter_12', trainable=block_1),
                               strides=[1, 1, 1, 1], padding='SAME', name='conv_12')
        prelu_12 = prelu(conv_12, name='prelu_12', trainable=block_1)

        conv_13 = tf.nn.conv2d(prelu_12,
                               weight_variable([3, 3, 64, 64], name='filter_13', trainable=block_1),
                               strides=[1, 1, 1, 1], padding='SAME', name='conv_13')
        prelu_13 = prelu(conv_13, name='res_act_12', trainable=block_1)
        res_1 = prelu_11 + prelu_13

        #
        conv_21 = tf.nn.conv2d(res_1,
                               weight_variable([3, 3, 64, 128], name='filter_21', trainable=block_2),
                               strides=[1, 2, 2, 1], padding='SAME', name='conv_21')
        prelu_21 = prelu(conv_21, name='prelu_21', trainable=block_2)

        conv_22 = tf.nn.conv2d(prelu_21,
                               weight_variable([3, 3, 128, 128], name='filter_22', trainable=block_2),
                               strides=[1, 1, 1, 1], padding='SAME', name='conv_22')
        prelu_22 = prelu(conv_22, name='prelu_22', trainable=block_2)

        conv_23 = tf.nn.conv2d(prelu_22,
                               weight_variable([3, 3, 128, 128], name='filter_23', trainable=block_2),
                               strides=[1, 1, 1, 1], padding='SAME', name='conv_23')
        prelu_23 = prelu(conv_23, name='prelu_23', trainable=block_2)
        res_21 = prelu_21 + prelu_23

        conv_24 = tf.nn.conv2d(res_21,
                                weight_variable([3, 3, 128, 128], name='filter_24', trainable=block_2),
                                   strides=[1, 1, 1, 1], padding='SAME', name='conv_24')
        prelu_24 = prelu(conv_24, name='prelu_24', trainable=block_2)

        conv_25 = tf.nn.conv2d(prelu_24,
                                   weight_variable([3, 3, 128, 128], name='filter_25', trainable=block_2),
                                   strides=[1, 1, 1, 1], padding='SAME', name='conv_25')
        prelu_25 = prelu(conv_25, name='prelu_25', trainable=block_2)
        res_22 = res_21 + prelu_25

        #
        conv_31 = tf.nn.conv2d(res_22,
                              weight_variable([3, 3, 128, 256], name='filter_31', trainable=block_3),
                              strides=[1, 2, 2, 1], padding='SAME', name='conv_31')
        prelu_31 = prelu(conv_31, name='prelu_31', trainable=block_3)

        conv_32 = tf.nn.conv2d(prelu_31,
                                   weight_variable([3, 3, 256, 256], name='filter_32', trainable=block_3),
                                   strides=[1, 1, 1, 1], padding='SAME', name='conv_32')
        ########
        prelu_32_1 = prelu(conv_32, name='prelu_32_1', trainable=block_3)
        conv_33_1 = tf.nn.conv2d(prelu_32_1,
                                   weight_variable([3, 3, 256, 256], name='filter_33_1', trainable=block_3),
                                   strides=[1, 1, 1, 1], padding='SAME', name='conv_33_1')
        prelu_33_1 = prelu(conv_33_1, name='prelu_33_1', trainable=block_3)
        res_31 = prelu_31 + prelu_33_1

        #######
        prelu_32_2 = prelu(conv_32, name='prelu_32_2', trainable=block_3)
        conv_33_2 = tf.nn.conv2d(prelu_32_2,
                                weight_variable([3, 3, 256, 256], name='filter_33_2', trainable=block_3),
                                strides=[1, 1, 1, 1], padding='SAME', name='conv_33_2')
        prelu_33_2 = prelu(conv_33_2, name='prelu_33_2', trainable=block_3)
        res_32 = res_31 + prelu_33_2

        #######
        prelu_32_3 = prelu(conv_32, name='prelu_32_3', trainable=block_3)
        conv_33_3 = tf.nn.conv2d(prelu_32_3,
                                   weight_variable([3, 3, 256, 256], name='filter_33_3', trainable=block_3),
                                   strides=[1, 1, 1, 1], padding='SAME', name='conv_33_3')
        prelu_33_3 = prelu(conv_33_3, name='prelu_33_3', trainable=block_3)
        res_33 = res_32 + prelu_33_3


        conv_34 = tf.nn.conv2d(res_33,
                                   weight_variable([3, 3, 256, 256], name='filter_34', trainable=block_3),
                                   strides=[1, 1, 1, 1], padding='SAME', name='conv_34')
        prelu_34 = prelu(conv_34, name='prelu_34', trainable=block_3)

        conv_35 = tf.nn.conv2d(prelu_34,
                                   weight_variable([3, 3, 256, 256], name='filter_35', trainable=block_3),
                                   strides=[1, 1, 1, 1], padding='SAME', name='conv_35')
        prelu_35 = prelu(conv_35, name='prelu_35', trainable=block_3)
        res_34 = res_33 + prelu_35

        #
        conv_41 = tf.nn.conv2d(res_34,
                              weight_variable([3, 3, 256, 512], name='filter_41', trainable=block_4),
                              strides=[1, 2, 2, 1], padding='SAME', name='conv_41')
        prelu_41 = prelu(conv_41, name='prelu_41', trainable=block_4)

        conv_42 = tf.nn.conv2d(prelu_41,
                                   weight_variable([3, 3, 512, 512], name='filter_42', trainable=block_4),
                                   strides=[1, 1, 1, 1], padding='SAME', name='conv_42')
        prelu_42 = prelu(conv_42, name='prelu_42', trainable=block_4)

        conv_43 = tf.nn.conv2d(prelu_42,
                                   weight_variable([3, 3, 512, 512], name='filter_43', trainable=block_4),
                                   strides=[1, 1, 1, 1], padding='SAME', name='conv_43')
        prelu_43 = prelu(conv_43, name='prelu_43', trainable=block_4)
        res_41 = prelu_41 + prelu_43

        flat1 = tf.layers.flatten(res_41, 'flat_1')
        fc_1 = tf.layers.dense(flat1, 512, name='fc_1', trainable=block_5)

        predict_logits = tf.layers.dense(fc_1, nb_classes, name='predict_logits', trainable=block_5)
        output_pred = tf.nn.softmax(predict_logits, name='output_pred', axis=1)
        loss = ce_loss(predict_logits, input_y)

        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss)
        correct = tf.equal(tf.cast(tf.argmax(output_pred, 1), tf.int32), tf.cast(tf.argmax(input_y, 1), tf.int32))
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    return train_op, acc, loss, output_pred


def gen_graph_stu(input_x, input_y, learning_rate, scope='student', reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as sc:
        flat1 = tf.layers.flatten(input_x, 'flat_1')
        fc_1 = tf.layers.dense(flat1, 512, name='fc1')
        predict_logits = tf.layers.dense(fc_1, nb_classes, name='predict_logits')
        output_pred = tf.nn.softmax(predict_logits, name='output_pred', axis=1)
        loss = ce_loss(predict_logits, input_y)
        correct = tf.equal(tf.cast(tf.argmax(output_pred, 1), tf.int32), tf.cast(tf.argmax(input_y, 1), tf.int32))
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    return loss, acc, output_pred


def train_distill():
    with tf.Graph().as_default() as g:
        input_x = tf.placeholder(tf.float32, [None, 112, 96, 3], name='input_x')
        input_y = tf.placeholder(tf.int32, [None, nb_classes], name='input_y')
        learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')
        train_op, acc_teacher, loss_teacher, y_pred_teacher = \
            gen_graph(input_x, input_y, learning_rate, scope="teacher")
        loss_student, acc_stu, y_pred_student = gen_graph_stu(input_x, input_y, learning_rate, )
        loss_student1 = tf.reduce_mean(- tf.reduce_sum(
            y_pred_teacher * tf.log(y_pred_student), reduction_indices=1)) + loss_student

        model_vars = tf.trainable_variables()
        var_teacher = [var for var in model_vars if 'teacher' in var.name]
        var_student = [var for var in model_vars if 'student' in var.name]
        grad_teacher = tf.gradients(loss_teacher, var_teacher)
        grad_student = tf.gradients(loss_student1, var_student)

        trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        trainer1 = tf.train.GradientDescentOptimizer(0.1)
        train_step_teacher = trainer.apply_gradients(zip(grad_teacher, var_teacher))
        train_step_student = trainer1.apply_gradients(zip(grad_student, var_student))

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver1 = tf.train.Saver(var_teacher)
        saver2 = tf.train.Saver(var_student)

        lr = 0.0001
        data_block = get_img_path_and_label(data_path, block_nums)
        train_block = data_block[: int(block_nums * 0.8)]
        val_block = data_block[int(block_nums * 0.8):]
        with open("train_images.txt", "w") as f:
            for i in train_block:
                for j in i[0]:
                    f.write(j.split("CASIA-WebFace-aligned")[-1] + "\n")

        with open("val_images.txt", "w") as f:
            for i in val_block:
                for j in i[0]:
                    f.write(j.split("CASIA-WebFace-aligned")[-1] + "\n")

        for epoch in range(0, 10):
            step = 0
            for block_idx in range(len(train_block)):
                block = train_block[block_idx]
                x, y = get_data(block)
                mini_batch = random_mini_batches(x, y, BATCH_SIZE)
                for batch in mini_batch:
                    x_batch, y_batch = batch
                    train_feed_dict = {input_x: (x_batch - 127.5) / 128, input_y: y_batch, learning_rate: lr}
                    if step % 50 == 0:
                        train_loss = loss_teacher.eval(
                            feed_dict=train_feed_dict)
                        train_accuracy = acc_teacher.eval(
                            feed_dict=train_feed_dict)
                        print("step %d, training loss %g , accuracy %g" % (step, train_loss, train_accuracy))
                    train_step_teacher.run(train_feed_dict)
                    step += 1


            # eval
            eval_acc_list = []
            for block_idx in range(len(val_block)):
                block = val_block[block_idx]
                x, y = get_data(block)
                mini_batch = random_mini_batches(x, y, BATCH_SIZE)
                for batch in mini_batch:
                    x_batch, y_batch = batch
                    val_feed_dict = {input_x: (x_batch - 127.5) / 128, input_y: y_batch, learning_rate: lr}
                    val_accuracy = acc_teacher.eval(
                        feed_dict=val_feed_dict)
                    eval_acc_list.append(val_accuracy)
            print('%d epoch eval acc: %f' % (epoch, sum(eval_acc_list) / len(eval_acc_list)))
            saver1.save(sess, './models/teacher.ckpt')

        for epoch in range(0, 100):
            step = 0
            for block_idx in range(len(train_block)):
                block = train_block[block_idx]
                x, y = get_data(block)
                mini_batch = random_mini_batches(x, y, BATCH_SIZE)
                for batch in mini_batch:
                    x_batch, y_batch = batch
                    train_feed_dict = {input_x: (x_batch - 127.5) / 128, input_y: y_batch, learning_rate: lr}
                    if step % 50 == 0:
                        train_loss = loss_student1.eval(
                            feed_dict=train_feed_dict)
                        train_accuracy = acc_stu.eval(
                            feed_dict=train_feed_dict)
                        print("step %d, training loss %g , accuracy %g" % (step, train_loss, train_accuracy))
                    train_step_student.run(train_feed_dict)
                    step += 1


            # eval
            eval_acc_list = []
            for block_idx in range(len(val_block)):
                block = val_block[block_idx]
                x, y = get_data(block)
                mini_batch = random_mini_batches(x, y, BATCH_SIZE)
                for batch in mini_batch:
                    x_batch, y_batch = batch
                    val_feed_dict = {input_x: (x_batch - 127.5) / 128, input_y: y_batch, learning_rate: lr}
                    val_accuracy = acc_stu.eval(
                        feed_dict=val_feed_dict)
                    eval_acc_list.append(val_accuracy)
            print('%d epoch eval acc: %f' % (epoch, sum(eval_acc_list) / len(eval_acc_list)))
            saver2.save(sess, './models/student.ckpt')


# local distill training
train_distill()

