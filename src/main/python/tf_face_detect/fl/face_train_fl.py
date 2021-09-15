import math
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image

nb_classes = 10575
BATCH_SIZE = 16
LambdaMin = 5.0
LambdaMax = 1500.0
load_check = False
data_path = '/home/zhangxun/data/CASIA-WebFace-aligned'
checkpoints_dir = 'checkpoints/'
block_nums = 80
client = 2


def prelu(input, name):
    alphas = tf.get_variable(name + '_alphas', input.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(input)
    neg = alphas * tf.nn.relu(-input)
    return tf.add(pos, neg, name=name)


def weight_variable(shape, stddev=0.2, name=None):
    initial = tf.glorot_uniform_initializer()
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, shape=shape, initializer=initial)


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


def gen_graph():
    weights = {
        'c1_1': weight_variable([3, 3, 3, 64], name='W_conv11'),
        'c1_2': weight_variable([3, 3, 64, 64], name='W_conv12'),
        'c1_3': weight_variable([3, 3, 64, 64], name='W_conv13'),

        'c2_1': weight_variable([3, 3, 64, 128], name='W_conv21'),
        'c2_2': weight_variable([3, 3, 128, 128], name='W_conv22'),
        'c2_3': weight_variable([3, 3, 128, 128], name='W_conv23'),
        'c2_4': weight_variable([3, 3, 128, 128], name='W_conv24'),
        'c2_5': weight_variable([3, 3, 128, 128], name='W_conv25'),

        'c3_1': weight_variable([3, 3, 128, 256], name='W_conv31'),
        'c3_2': weight_variable([3, 3, 256, 256], name='W_conv32'),
        'c3_3': weight_variable([3, 3, 256, 256], name='W_conv33'),
        'c3_4': weight_variable([3, 3, 256, 256], name='W_conv34'),
        'c3_5': weight_variable([3, 3, 256, 256], name='W_conv35'),
        'c3_6': weight_variable([3, 3, 256, 256], name='W_conv36'),
        'c3_7': weight_variable([3, 3, 256, 256], name='W_conv37'),
        'c3_8': weight_variable([3, 3, 256, 256], name='W_conv38'),
        'c3_9': weight_variable([3, 3, 256, 256], name='W_conv39'),

        'c4_1': weight_variable([3, 3, 256, 512], name='W_conv41'),
        'c4_2': weight_variable([3, 3, 512, 512], name='W_conv42'),
        'c4_3': weight_variable([3, 3, 512, 512], name='W_conv43'),

        'fc5': weight_variable([512 * 7 * 6, 512], name='W_fc5'),
    }

    # global_step = tf.Variable(0, trainable=False)
    input_x = tf.placeholder(tf.float32, [None] + [112, 96] + [3], name='input_x')
    input_y = tf.placeholder(tf.int32, [None, nb_classes], name='input_y')
    learning_rate = tf.placeholder(tf.float32, (), name='lr')
    # stp = tf.placeholder(tf.float32, (), name='stp')

    conv11 = tf.nn.conv2d(input_x, weights['c1_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_11')
    h_1 = prelu(conv11, name='act_1')

    # ResNet-1
    res_conv_11 = tf.nn.conv2d(h_1, weights['c1_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_12')
    res_h_11 = prelu(res_conv_11, name='res_act_11')
    res_conv_12 = tf.nn.conv2d(res_h_11, weights['c1_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_13')
    res_h_12 = prelu(res_conv_12, name='res_act_12')
    res_1 = h_1 + res_h_12

    # ResNet-2
    conv21 = tf.nn.conv2d(res_1, weights['c2_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_21')
    h_2 = prelu(conv21, name='act_2')
    res_conv_21 = tf.nn.conv2d(h_2, weights['c2_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_22')
    res_h_21 = prelu(res_conv_21, name='res_act_21')
    res_conv_22 = tf.nn.conv2d(res_h_21, weights['c2_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_23')
    res_h_22 = prelu(res_conv_22, name='res_act_22')
    res_21 = h_2 + res_h_22

    res_conv_23 = tf.nn.conv2d(res_21, weights['c2_4'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_24')
    res_h_23 = prelu(res_conv_23, name='res_act_23')
    res_conv_24 = tf.nn.conv2d(res_h_23, weights['c2_5'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_25')
    res_h_24 = prelu(res_conv_24, name='res_act_24')
    res_22 = res_21 + res_h_24

    # ResNet-3
    conv31 = tf.nn.conv2d(res_22, weights['c3_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_31')
    h_3 = prelu(conv31, name='act_3')
    res_conv_31 = tf.nn.conv2d(h_3, weights['c3_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_31')
    res_h_31 = prelu(res_conv_31, name='res_act_31')
    res_conv_32 = tf.nn.conv2d(res_h_31, weights['c3_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_32')
    res_h_32 = prelu(res_conv_32, name='res_act_32')
    res_31 = h_3 + res_h_32

    # res_conv_33 = tf.nn.conv2d(res_31, weights['c3_4'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_33')
    res_h_33 = prelu(res_conv_31, name='res_act_33')
    res_conv_34 = tf.nn.conv2d(res_h_33, weights['c3_5'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_34')
    res_h_34 = prelu(res_conv_34, name='res_act_34')
    res_32 = res_31 + res_h_34

    # res_conv_35 = tf.nn.conv2d(res_32, weights['c3_6'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_35')
    res_h_35 = prelu(res_conv_31, name='res_act_35')
    res_conv_36 = tf.nn.conv2d(res_h_35, weights['c3_7'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_36')
    res_h_36 = prelu(res_conv_36, name='res_act_36')
    res_33 = res_32 + res_h_36

    res_conv_37 = tf.nn.conv2d(res_33, weights['c3_8'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_37')
    res_h_37 = prelu(res_conv_37, name='res_act_37')
    res_conv_38 = tf.nn.conv2d(res_h_37, weights['c3_9'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_38')
    res_h_38 = prelu(res_conv_38, name='res_act_38')
    res_34 = res_33 + res_h_38

    # ResNet-4
    conv41 = tf.nn.conv2d(res_34, weights['c4_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_41')
    h_4 = prelu(conv41, name='act_4')
    res_conv_41 = tf.nn.conv2d(h_4, weights['c4_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_41')
    res_h_41 = prelu(res_conv_41, name='res_act_41')
    res_conv_42 = tf.nn.conv2d(res_h_41, weights['c4_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_42')
    res_h_42 = prelu(res_conv_42, name='res_act_42')
    res_41 = h_4 + res_h_42

    flat1 = tf.layers.flatten(res_41, 'flat_1')
    fc_1 = tf.layers.dense(flat1, 512, name='fc_1')

    logits = tf.layers.dense(fc_1, nb_classes, name='pred_logits')
    output_pred = tf.nn.softmax(logits, name='output', axis=1)
    loss = ce_loss(logits, input_y)

    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss)
    correct = tf.equal(tf.cast(tf.argmax(output_pred, 1), tf.int32), tf.cast(tf.argmax(input_y, 1), tf.int32))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    return input_x, input_y, learning_rate, train_op, loss, acc


def train():
    data_block = get_img_path_and_label(data_path, block_nums)
    train_block = data_block[: int(block_nums*0.8)]
    val_block = data_block[int(block_nums*0.8):]
    with open("train_images.txt", "w") as f:
        for i in train_block:
            for j in i[0]:
                f.write(j.split("CASIA-WebFace-aligned")[-1] + "\n")
    block_per = len(train_block) // client
    lr = 0.0001
    clients_weights = [[] for _ in range(client)]
    update_weights = [[] for _ in range(client)]
    for epoch in range(0, 100):
        tf.contrib.keras.backend.clear_session()
        with tf.Graph().as_default():
            input_x, input_y, learning_rate, train_op, loss, acc = gen_graph()
            saver = tf.train.Saver(max_to_keep=5)
            with tf.Session() as sess:
                if load_check:
                    checkpoints = tf.train.get_checkpoint_state(checkpoints_dir)
                    meta_graph_path = checkpoints.model_checkpoint_path + '.meta'
                    restore = tf.train.import_meta_graph(meta_graph_path)
                    restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
                    saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
                else:
                    sess.run(tf.global_variables_initializer())
                for i in range(client):
                    step = 0
                    if epoch == 0:
                        sess.run(tf.global_variables_initializer())
                    else:
                        trainable_var = tf.trainable_variables()
                        for var, c_w in zip(trainable_var,zip(*update_weights)):
                            mean_val = 0
                            for j in c_w:
                                mean_val = mean_val + j
                            mean_val = mean_val / len(c_w)
                            mean_val_op = var.assign(mean_val)
                            sess.run(mean_val_op)
                    train_local_block = train_block[i*block_per: (i+1)*block_per]

                    with open("train_images_"+str(i)+".txt", "w") as f:
                        for train_local_block_i in train_local_block:
                            for train_local_block_i_j in train_local_block_i[0]:
                                f.write(train_local_block_i_j.split("CASIA-WebFace-aligned")[-1] + "\n")

                    for block_idx in range(len(train_local_block)):
                        block = train_local_block[block_idx]
                        x, y = get_data(block)
                        mini_batch = random_mini_batches(x, y, BATCH_SIZE)
                        for batch in mini_batch:
                            x_batch, y_batch = batch
                            train_feed_dict = {input_x: (x_batch - 127.5) / 128, input_y: y_batch, learning_rate: lr}
                            _, train_loss, train_acc = sess.run([train_op, loss, acc], train_feed_dict)
                            step += 1
                            if step % 50 == 0:
                                pre_loss = train_loss
                                with open("client"+str(i)+".txt", "a+") as f:
                                    f.write('%d epoch %d step train loss : %f acc: %f' % (epoch, step, train_loss, train_acc))
                                    f.write("\n")
                    # eval
                    eval_loss_list = []
                    eval_acc_list = []
                    block_val_per = len(val_block) // client
                    val_local_block = val_block[i * block_val_per: (i + 1) * block_val_per]

                    with open("val_images_"+str(i)+".txt", "w") as f:
                        for val_local_block_i in val_local_block:
                            for val_local_block_i_j in val_local_block_i[0]:
                                f.write(val_local_block_i_j.split("CASIA-WebFace-aligned")[-1] + "\n")

                    for block_idx in range(len(val_local_block)):
                        block = val_local_block[block_idx]
                        x, y = get_data(block)
                        mini_batch = random_mini_batches(x, y, BATCH_SIZE)
                        for batch in mini_batch:
                            x_batch, y_batch = batch
                            val_feed_dict = {input_x: (x_batch - 127.5) / 128, input_y: y_batch, learning_rate: lr}
                            eval_loss, eval_acc = sess.run([loss, acc], feed_dict=val_feed_dict)
                            eval_loss_list.append(eval_loss)
                            eval_acc_list.append(eval_acc)
                    with open("client" + str(i) + ".txt", "a+") as f:
                        f.write('%d epoch eval loss : %f acc: %f' % (epoch, sum(eval_loss_list) / len(eval_loss_list),
                                                                            sum(eval_acc_list)/len(eval_acc_list)))
                        f.write("\n")
                    trainable_var = tf.trainable_variables()
                    c = []
                    for var in trainable_var:
                        tmp = sess.run(var)
                        c.append(tmp)
                    clients_weights[i] = c
                for idx, weights_cli in enumerate(clients_weights):
                    update_weights[idx] = weights_cli


train()

