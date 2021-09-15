import math
import os
import random
import json
import argparse

from google.protobuf import json_format
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import mxnet as mx
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
import cv2
from scipy.spatial import distance


parser = argparse.ArgumentParser(description='frozen_layers to this script')
parser.add_argument('--unfrozen', type=int, default=None)
parser.add_argument('-p', '--path', help='the root path ')
args = parser.parse_args()

path = args.path
block_idx = args.unfrozen

path = "./model_info/"

val_data = "/Users/voyager/Downloads/MobileFaceNet_TF-master/datasets/faces_ms1m_112x112"

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

nb_classes = 1006
BATCH_SIZE = 64
LambdaMin = 5.0
LambdaMax = 1500.0
load_check = False
data_path = '/Users/voyager/Downloads/facenet/data/CASIA-WebFace-aligned'
checkpoints_dir = 'checkpoints/'
block_nums = 80
generated_graph = False


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


def load_val_data(db_name, image_size):
    bins, issame_list = pickle.load(open(os.path.join(val_data, db_name + '.bin'), 'rb'), encoding='bytes')
    datasets = np.empty((len(issame_list) * 2, image_size[0], image_size[1], 3))

    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()

        # img = cv2.imdecode(np.fromstring(_bin, np.uint8), -1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img - 127.5
        img = img * 0.0078125
        img = cv2.resize(img, dsize=(image_size[1], image_size[0]))
        datasets[i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(datasets.shape)

    return datasets, issame_list

max_threshold = 0
min_threshold = 4


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    global max_threshold
    global min_threshold

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        # print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

        if max_threshold < thresholds[best_threshold_index]:
            max_threshold = thresholds[best_threshold_index]
        if min_threshold > thresholds[best_threshold_index]:
            min_threshold = thresholds[best_threshold_index]
    print('thresholds max: {} <=> min: {}'.format(max_threshold, min_threshold))

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
    return tpr, fpr, accuracy


with tf.Graph().as_default() as g:
    weights = {
        'c1_1': weight_variable([3, 3, 3, 64], name='W_conv11', trainable=block_1),
        'c1_2': weight_variable([3, 3, 64, 64], name='W_conv12', trainable=block_1),
        'c1_3': weight_variable([3, 3, 64, 64], name='W_conv13', trainable=block_1),

        'c2_1': weight_variable([3, 3, 64, 128], name='W_conv21', trainable=block_2),
        'c2_2': weight_variable([3, 3, 128, 128], name='W_conv22', trainable=block_2),
        'c2_3': weight_variable([3, 3, 128, 128], name='W_conv23', trainable=block_2),
        'c2_4': weight_variable([3, 3, 128, 128], name='W_conv24', trainable=block_2),
        'c2_5': weight_variable([3, 3, 128, 128], name='W_conv25', trainable=block_2),

        'c3_1': weight_variable([3, 3, 128, 256], name='W_conv31', trainable=block_3),
        'c3_2': weight_variable([3, 3, 256, 256], name='W_conv32', trainable=block_3),
        'c3_3': weight_variable([3, 3, 256, 256], name='W_conv33', trainable=block_3),
        'c3_4': weight_variable([3, 3, 256, 256], name='W_conv34', trainable=block_3),
        'c3_5': weight_variable([3, 3, 256, 256], name='W_conv35', trainable=block_3),
        'c3_6': weight_variable([3, 3, 256, 256], name='W_conv36', trainable=block_3),
        'c3_7': weight_variable([3, 3, 256, 256], name='W_conv37', trainable=block_3),
        'c3_8': weight_variable([3, 3, 256, 256], name='W_conv38', trainable=block_3),
        'c3_9': weight_variable([3, 3, 256, 256], name='W_conv39', trainable=block_3),

        'c4_1': weight_variable([3, 3, 256, 512], name='W_conv41', trainable=block_4),
        'c4_2': weight_variable([3, 3, 512, 512], name='W_conv42', trainable=block_4),
        'c4_3': weight_variable([3, 3, 512, 512], name='W_conv43', trainable=block_4),

        'fc5': weight_variable([512 * 7 * 6, 512], name='W_fc5',  trainable=block_5),
    }

    global_step = tf.Variable(0, trainable=False)
    input_x = tf.placeholder(tf.float32, [None] + [112, 96] + [3], name='input_x')
    input_y = tf.placeholder(tf.int32, [None, nb_classes], name='input_y')
    learning_rate = tf.placeholder(tf.float32, (), name='lr')
    stp = tf.placeholder(tf.float32, (), name='stp')

    conv11 = tf.nn.conv2d(input_x, weights['c1_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_11')
    h_1 = prelu(conv11, name='act_1', trainable=block_1)

    # ResNet-1
    res_conv_11 = tf.nn.conv2d(h_1, weights['c1_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_12')
    res_h_11 = prelu(res_conv_11, name='res_act_11', trainable=block_1)
    res_conv_12 = tf.nn.conv2d(res_h_11, weights['c1_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_13')
    res_h_12 = prelu(res_conv_12, name='res_act_12', trainable=block_1)
    res_1 = h_1 + res_h_12

    # ResNet-2
    conv21 = tf.nn.conv2d(res_1, weights['c2_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_21')
    h_2 = prelu(conv21, name='act_2', trainable=block_2)
    res_conv_21 = tf.nn.conv2d(h_2, weights['c2_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_22')
    res_h_21 = prelu(res_conv_21, name='res_act_21', trainable=block_2)
    res_conv_22 = tf.nn.conv2d(res_h_21, weights['c2_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_23')
    res_h_22 = prelu(res_conv_22, name='res_act_22', trainable=block_2)
    res_21 = h_2 + res_h_22

    res_conv_23 = tf.nn.conv2d(res_21, weights['c2_4'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_24')
    res_h_23 = prelu(res_conv_23, name='res_act_23', trainable=block_2)
    res_conv_24 = tf.nn.conv2d(res_h_23, weights['c2_5'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_25')
    res_h_24 = prelu(res_conv_24, name='res_act_24', trainable=block_2)
    res_22 = res_21 + res_h_24

    # ResNet-3
    conv31 = tf.nn.conv2d(res_22, weights['c3_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_31')
    h_3 = prelu(conv31, name='act_3', trainable=block_3)
    res_conv_31 = tf.nn.conv2d(h_3, weights['c3_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_31')
    res_h_31 = prelu(res_conv_31, name='res_act_31', trainable=block_3)
    res_conv_32 = tf.nn.conv2d(res_h_31, weights['c3_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_32')
    res_h_32 = prelu(res_conv_32, name='res_act_32', trainable=block_3)
    res_31 = h_3 + res_h_32

    res_conv_33 = tf.nn.conv2d(res_31, weights['c3_4'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_33')
    res_h_33 = prelu(res_conv_31, name='res_act_33', trainable=block_3)
    res_conv_34 = tf.nn.conv2d(res_h_33, weights['c3_5'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_34')
    res_h_34 = prelu(res_conv_34, name='res_act_34', trainable=block_3)
    res_32 = res_31 + res_h_34

    res_conv_35 = tf.nn.conv2d(res_32, weights['c3_6'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_35')
    res_h_35 = prelu(res_conv_31, name='res_act_35', trainable=block_3)
    res_conv_36 = tf.nn.conv2d(res_h_35, weights['c3_7'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_36')
    res_h_36 = prelu(res_conv_36, name='res_act_36', trainable=block_3)
    res_33 = res_32 + res_h_36

    res_conv_37 = tf.nn.conv2d(res_33, weights['c3_8'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_37')
    res_h_37 = prelu(res_conv_37, name='res_act_37', trainable=block_3)
    res_conv_38 = tf.nn.conv2d(res_h_37, weights['c3_9'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_38')
    res_h_38 = prelu(res_conv_38, name='res_act_38', trainable=block_3)
    res_34 = res_33 + res_h_38

    # ResNet-4
    conv41 = tf.nn.conv2d(res_34, weights['c4_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_41')
    h_4 = prelu(conv41, name='act_4', trainable=block_4)
    res_conv_41 = tf.nn.conv2d(h_4, weights['c4_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_41')
    res_h_41 = prelu(res_conv_41, name='res_act_41', trainable=block_4)
    res_conv_42 = tf.nn.conv2d(res_h_41, weights['c4_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_42')
    res_h_42 = prelu(res_conv_42, name='res_act_42', trainable=block_4)
    res_41 = h_4 + res_h_42

    flat1 = tf.layers.flatten(res_41, 'flat_1')
    fc_1 = tf.layers.dense(flat1, 512, name='fc_1', trainable=block_5)

    logits = tf.layers.dense(fc_1, nb_classes, name='pred_logits', trainable=block_5)
    output_pred = tf.nn.softmax(logits, name='output', axis=1)
    loss = ce_loss(logits, input_y)

    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss)
    correct = tf.equal(tf.cast(tf.argmax(output_pred, 1), tf.int32), tf.cast(tf.argmax(input_y, 1), tf.int32))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    if generated_graph:
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        graph_def = sess.graph.as_graph_def()
        json_string = json_format.MessageToJson(graph_def)
        obj = json.loads(json_string)
        tf.compat.v1.train.write_graph(sess.graph, path, 'sphere_unfrozen' + '.pb', as_text=False)

        # generate txt

        trainable_var = tf.trainable_variables()
        global_var = tf.global_variables()
        with open(path + "sphere2_trainable_var_unfrozen" + ".txt", "w") as f:
            variables_sum = 0
            for var in trainable_var:
                accumulate = 1
                for i in range(len(var.shape)):
                    accumulate = var.shape[i] * accumulate
                variables_sum = accumulate + variables_sum
                f.write(var.initial_value.op.name + ";" + str(var.op.name) + "\n")
            print(variables_sum)

        with open(path + "sphere2_trainable_init_var_unfrozen" + ".txt", "w") as f:
            variables_sum = 0
            for var in global_var:
                accumulate = 1
                for i in range(len(var.shape)):
                    accumulate = var.shape[i] * accumulate
                variables_sum = accumulate + variables_sum
                if 'Momentum' not in var.initial_value.op.name:
                    f.write(var.initial_value.op.name + ";" + str(var.shape) + "\n")
            print(variables_sum)

        with open(path + "sphere2_feed_fetch_unfrozen" + ".txt", "w") as f:
            f.write(input_y.op.name + ";" + str(input_y.shape) + "\n")
            f.write(input_x.op.name + ";" + str(input_x.shape) + "\n")
            f.write(init.name + ";" + "---" + "\n")
            f.write(train_op.name + ";" + "---" + "\n")
            f.write(loss.name + ";" + "---" + "\n")
            f.write(learning_rate.name + ";" + "---" + "\n")
            f.write(acc.name + ";" + "---" + "\n")
    else:
        data_block = get_img_path_and_label(data_path, block_nums)
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
            step = 0
            lr = 0.0001
            stop = False
            for epoch in range(0, 100):
                step = 0
                for block_idx in range(len(data_block)):
                    block = data_block[block_idx]
                    x, y = get_data(block)
                    mini_batch = random_mini_batches(x, y, BATCH_SIZE)
                    for batch in mini_batch:
                        x_batch, y_batch = batch
                        train_feed_dict = {input_x: (x_batch - 127.5) / 128, input_y: y_batch, learning_rate: lr}
                        _, train_loss, train_acc = sess.run([train_op, loss, acc], train_feed_dict)
                        step += 1
                        if step % 50 == 0:
                            pre_loss = train_loss
                            print('%d epoch %d step train loss : %f acc: %f' % (epoch, step, train_loss, train_acc))
                        if step % 2000 == 0:
                            path = saver.save(sess, checkpoints_dir + '/model.ckpt', global_step=step)
                            print('Save model in step:{}, path:{}'.format(step, path))
                # eval use lfw

                ver_list = []
                ver_name_list = []
                data_sets, issame_list = load_val_data("lfw", [112, 96])
                cos_list = []
                acc_num = 0
                for i in range(120):
                    ds = data_sets[i*100: (i+1)*100]
                    issame_list_sub = issame_list[i*50:(i+1)*50]
                    val_feed_dict = {input_x: ds}
                    val_em = sess.run([fc_1], feed_dict=val_feed_dict)
                    embeddings1 = val_em[0][0::2]
                    embeddings2 = val_em[0][1::2]
                    for i in range(len(embeddings2)):
                        tmp = distance.cosine(embeddings1[i], embeddings2[i])
                        if tmp >= 0.8 and issame_list_sub[i]:
                            acc_num +=1

                print(acc_num/len(data_sets))
