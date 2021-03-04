import tensorflow as tf
import numpy as np
import os
from PIL import Image
import math
import random
import pickle
from sklearn import metrics
from sklearn.model_selection import KFold
import mxnet as mx
import cv2

nb_classes = 1006
NB_CLASS = 10575
BATCH_SIZE = 64
LambdaMin = 5.0
LambdaMax = 1500.0
val_data = "/home/zhangxun/MobileFaceNet_TF-master/datasets/faces_ms1m_112x112"

def prelu(input, name):
    alphas = tf.get_variable(name + '_alphas', input.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(input)
    neg = alphas * tf.nn.relu(-input)
    return tf.add(pos, neg, name=name)


def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)


def add_to_regularizetion_loss(W, b=None):
    tf.add_to_collection('losses', tf.nn.l2_loss(W))
    if b:
        tf.add_to_collection('losses', tf.nn.l2_loss(b))


def weight_variable(shape, stddev=0.2, name=None):
    # initial = tf.truncated_normal(shape, stddev=stddev)
    initial = tf.glorot_uniform_initializer()
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, shape=shape, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.glorot_uniform_initializer()
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, shape=shape, initializer=initial)


def ce_loss(logit, label, reg_ratio=0.):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=label))
    return cross_entropy_loss




def center_loss(features, label, alfa, nb_classes):
    # label: (N, nb_classes), features: (N, embed_dim), centers: (nb_classes, embed_dim)
    embed_dim = features.get_shape()[1]
    centers = tf.get_variable('centers', [nb_classes, embed_dim], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # label_n = np.cast(np.array(label), dtype=np.int32)

    label = tf.cast(label, tf.float32)
    # label = np.eye(nb_classes, dtype=np.float32)[label]

    diff = tf.matmul(label, tf.matmul(label, centers) - features, transpose_a=True)
    center_count = tf.reduce_sum(tf.transpose(label), axis=1, keepdims=True) + 1
    diff = diff / center_count
    centers = centers - alfa * diff
    with tf.control_dependencies([centers]):
        res = tf.reduce_sum(features - tf.matmul(label, centers), axis=1)
        loss = tf.reduce_mean(tf.square(res))
    return loss, centers


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    # shuffled_Y = Y[permutation, :]
    shuffled_Y = Y[permutation]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        # mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
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
    # print(image_cate)
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
    return data_block



# #
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

#########################val################
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

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    print("dist:"+str(max(dist))+":"+str(min(dist)))
    global max_threshold
    global min_threshold

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
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


with tf.Graph().as_default():
    weights = {
        'c1_1': weight_variable([3, 3, 3, 64], name='W_conv11'),
        'c1_2': weight_variable([3, 3, 64, 64], name='W_conv12'),
        'c1_3': weight_variable([3, 3, 64, 64], name='W_conv13'),
        'b1_1': bias_variable([64], name='b_conv11'),
        'b1_2': bias_variable([64], name='b_conv12'),
        'b1_3': bias_variable([64], name='b_conv13'),

        'c2_1': weight_variable([3, 3, 64, 128], name='W_conv21'),
        'c2_2': weight_variable([3, 3, 128, 128], name='W_conv22'),
        'c2_3': weight_variable([3, 3, 128, 128], name='W_conv23'),
        'c2_4': weight_variable([3, 3, 128, 128], name='W_conv24'),
        'c2_5': weight_variable([3, 3, 128, 128], name='W_conv25'),
        'b2_1': bias_variable([128], name='b_conv21'),
        'b2_2': bias_variable([128], name='b_conv22'),
        'b2_3': bias_variable([128], name='b_conv23'),
        'b2_4': bias_variable([128], name='b_conv24'),
        'b2_5': bias_variable([128], name='b_conv25'),

        'c3_1': weight_variable([3, 3, 128, 256], name='W_conv31'),
        'c3_2': weight_variable([3, 3, 256, 256], name='W_conv32'),
        'c3_3': weight_variable([3, 3, 256, 256], name='W_conv33'),
        'c3_4': weight_variable([3, 3, 256, 256], name='W_conv34'),
        'c3_5': weight_variable([3, 3, 256, 256], name='W_conv35'),
        'c3_6': weight_variable([3, 3, 256, 256], name='W_conv36'),
        'c3_7': weight_variable([3, 3, 256, 256], name='W_conv37'),
        'c3_8': weight_variable([3, 3, 256, 256], name='W_conv38'),
        'c3_9': weight_variable([3, 3, 256, 256], name='W_conv39'),
        'b3_1': bias_variable([256], name='b_conv31'),
        'b3_2': bias_variable([256], name='b_conv32'),
        'b3_3': bias_variable([256], name='b_conv33'),
        'b3_4': bias_variable([256], name='b_conv34'),
        'b3_5': bias_variable([256], name='b_conv35'),
        'b3_6': bias_variable([256], name='b_conv36'),
        'b3_7': bias_variable([256], name='b_conv37'),
        'b3_8': bias_variable([256], name='b_conv38'),
        'b3_9': bias_variable([256], name='b_conv39'),

        'c4_1': weight_variable([3, 3, 256, 512], name='W_conv41'),
        'c4_2': weight_variable([3, 3, 512, 512], name='W_conv42'),
        'c4_3': weight_variable([3, 3, 512, 512], name='W_conv43'),
        'b4_1': bias_variable([512], name='b_conv41'),
        'b4_2': bias_variable([512], name='b_conv42'),
        'b4_3': bias_variable([512], name='b_conv43'),

        'fc5': weight_variable([512 * 7 * 6, 512], name='W_fc5'),
        'b5': bias_variable([512], name='b_fc5'),
    }

    global_step = tf.Variable(0, trainable=False)
    input_x = tf.placeholder(tf.float32, [None] + [112, 96] + [3], name='input_x')
    input_y = tf.placeholder(tf.float32, [None, nb_classes], name='input_y')
    learning_rate = tf.placeholder(tf.float32, (), name='lr')

    conv11 = tf.nn.conv2d(input_x, weights['c1_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_11')
    h_conv11 = tf.nn.bias_add(conv11, weights['b1_1'])
    h_1 = prelu(h_conv11, name='act_1')

    # ResNet-1
    res_conv_11 = tf.nn.conv2d(h_1, weights['c1_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_12')
    h_res_conv_11 = tf.nn.bias_add(res_conv_11, weights['b1_2'])
    res_h_11 = prelu(h_res_conv_11, name='res_act_11')
    res_conv_12 = tf.nn.conv2d(res_h_11, weights['c1_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_13')
    h_res_conv_12 = tf.nn.bias_add(res_conv_12, weights['b1_3'])
    res_h_12 = prelu(h_res_conv_12, name='res_act_12')
    res_1 = h_1 + res_h_12

    # ResNet-2
    conv21 = tf.nn.conv2d(res_1, weights['c2_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_21')
    h_conv21 = tf.nn.bias_add(conv21, weights['b2_1'])
    h_2 = prelu(h_conv21, name='act_2')
    res_conv_21 = tf.nn.conv2d(h_2, weights['c2_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_22')
    h_res_conv_21 = tf.nn.bias_add(res_conv_21, weights['b2_2'])
    res_h_21 = prelu(h_res_conv_21, name='res_act_21')
    res_conv_22 = tf.nn.conv2d(res_h_21, weights['c2_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_23')
    h_res_conv_22 = tf.nn.bias_add(res_conv_22, weights['b2_3'])
    res_h_22 = prelu(h_res_conv_22, name='res_act_22')
    res_21 = h_2 + res_h_22

    res_conv_23 = tf.nn.conv2d(res_21, weights['c2_4'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_24')
    h_res_conv_23 = tf.nn.bias_add(res_conv_23, weights['b2_4'])
    res_h_23 = prelu(h_res_conv_23, name='res_act_23')
    res_conv_24 = tf.nn.conv2d(res_h_23, weights['c2_5'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_25')
    h_res_conv_24 = tf.nn.bias_add(res_conv_24, weights['b2_5'])
    res_h_24 = prelu(h_res_conv_24, name='res_act_24')
    res_22 = res_21 + res_h_24

    # ResNet-3
    conv31 = tf.nn.conv2d(res_22, weights['c3_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_31')
    h_conv31 = tf.nn.bias_add(conv31, weights['b3_1'])
    h_3 = prelu(h_conv31, name='act_3')
    res_conv_31 = tf.nn.conv2d(h_3, weights['c3_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_31')
    h_res_conv_31 = tf.nn.bias_add(res_conv_31, weights['b3_2'])
    res_h_31 = prelu(h_res_conv_31, name='res_act_31')
    res_conv_32 = tf.nn.conv2d(res_h_31, weights['c3_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_32')
    h_res_conv_32 = tf.nn.bias_add(res_conv_32, weights['b3_3'])
    res_h_32 = prelu(h_res_conv_32, name='res_act_32')
    res_31 = h_3 + res_h_32

    res_conv_33 = tf.nn.conv2d(res_31, weights['c3_4'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_33')
    h_res_conv_33 = tf.nn.bias_add(res_conv_33, weights['b3_4'])
    res_h_33 = prelu(h_res_conv_33, name='res_act_33')
    res_conv_34 = tf.nn.conv2d(res_h_33, weights['c3_5'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_34')
    h_res_conv_34 = tf.nn.bias_add(res_conv_34, weights['b3_5'])
    res_h_34 = prelu(h_res_conv_34, name='res_act_34')
    res_32 = res_31 + res_h_34

    res_conv_35 = tf.nn.conv2d(res_32, weights['c3_6'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_35')
    h_res_conv_35 = tf.nn.bias_add(res_conv_35, weights['b3_6'])
    res_h_35 = prelu(h_res_conv_35, name='res_act_35')
    res_conv_36 = tf.nn.conv2d(res_h_35, weights['c3_7'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_36')
    h_res_conv_36 = tf.nn.bias_add(res_conv_36, weights['b3_7'])
    res_h_36 = prelu(h_res_conv_36, name='res_act_36')
    res_33 = res_32 + res_h_36

    res_conv_37 = tf.nn.conv2d(res_33, weights['c3_8'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_37')
    h_res_conv_37 = tf.nn.bias_add(res_conv_37, weights['b3_8'])
    res_h_37 = prelu(h_res_conv_37, name='res_act_37')
    res_conv_38 = tf.nn.conv2d(res_h_37, weights['c3_9'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_38')
    h_res_conv_38 = tf.nn.bias_add(res_conv_38, weights['b3_9'])
    res_h_38 = prelu(h_res_conv_38, name='res_act_38')
    res_34 = res_33 + res_h_38

    # ResNet-4
    conv41 = tf.nn.conv2d(res_34, weights['c4_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_41')
    h_conv41 = tf.nn.bias_add(conv41, weights['b4_1'])
    h_4 = prelu(h_conv41, name='act_4')
    res_conv_41 = tf.nn.conv2d(h_4, weights['c4_2'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_41')
    h_res_conv_41 = tf.nn.bias_add(res_conv_41, weights['b4_2'])
    res_h_41 = prelu(h_res_conv_41, name='res_act_41')
    res_conv_42 = tf.nn.conv2d(res_h_41, weights['c4_3'], strides=[1, 1, 1, 1], padding='SAME', name='res_c_42')
    h_res_conv_42 = tf.nn.bias_add(res_conv_42, weights['b4_3'])
    res_h_42 = prelu(h_res_conv_42, name='res_act_42')
    res_41 = h_4 + res_h_42

    flat1 = tf.layers.flatten(res_41, 'flat_1')
    fc_1 = tf.matmul(flat1, weights['fc5']) + weights['b5']
    fc_1_embeddings = tf.nn.l2_normalize(fc_1, 1, 1e-10, name='embeddings')

    logits = tf.layers.dense(fc_1, nb_classes, name='pred_logits')
    output_pred = tf.nn.softmax(logits, name='output', axis=1)
    loss_val = ce_loss(logits, input_y)

    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss_val)
    correct = tf.equal(tf.argmax(output_pred, 1), tf.argmax(input_y, 1))
    # correct = tf.equal(tf.cast(tf.argmax(output_pred, 1), tf.int32), input_y)
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    # x, y, x_eval, y_eval, nb_classes = load_data('/home/zhangziyang/facial_recong/CASIA-WebFace-aligned')  # 路径改一下
    # minibatch = random_mini_batches(x, y, mini_batch_size=64)

    data_block = get_img_path_and_label('/home/zhangxun/data/CASIA-WebFace-aligned', 80)
    print(data_block)
    # checkpoints_dir = None
    # load_model = None

    # if load_model is None:
    #     current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    #     checkpoints_dir = 'checkpoints_4/{}'.format(current_time)
    #     print(checkpoints_dir)
    #     try:
    #         os.makedirs(checkpoints_dir)
    #     except os.error:
    #         pass
    # else:
    checkpoints_dir = 'checkpoints/'
    print(checkpoints_dir)

    # print(x.shape, x_eval.shape)
    # print(y.shape, y_eval.shape)
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        # if load_model:
        #     print(checkpoints_dir)
        #     checkpoints = tf.train.get_checkpoint_state(checkpoints_dir)
        #     meta_graph_path = checkpoints.model_checkpoint_path + '.meta'
        #     restore = tf.train.import_meta_graph(meta_graph_path)
        #     restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
        #     saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
        #     print(tf.train.latest_checkpoint(checkpoints_dir))
        #     step = int(meta_graph_path.split('-')[2].split('.')[0])
        # else:
        sess.run(tf.global_variables_initializer())
        step = 0

        # converter = tf.lite.TFLiteConverter.from_session(sess, [input_x], [embed])
        # tflite_model = converter.convert()

        # with open('model_train.tflite','wb') as f:
        # 	f.write(tflite_model)

        # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        pre_loss = 1000000
        lr = 0.0001
        stop = False
        data_sets, issame_list = load_val_data("lfw", [112, 96])
        for epoch in range(0, 100):
            step = 0
            print('IN EPOCH {}'.format(epoch))
            # eval
            val_em_all = np.zeros((12000, 512))
            for i in range(120):
                ds = data_sets[i * 100: (i + 1) * 100]
                val_feed_dict = {input_x: ds}
                val_em = sess.run([fc_1_embeddings], feed_dict=val_feed_dict)
                val_em_all[i * 100: (i + 1) * 100] = val_em[0]
            tpr, fpr, accuracy = evaluate(val_em_all, issame_list,
                                          nrof_folds=10)
            auc = metrics.auc(fpr, tpr)
            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))
            print('Area Under Curve (AUC): %1.3f' % auc)

            for block_idx in range(len(data_block)):
                block = data_block[block_idx]
                x, y = get_data(block)
                minibatch = random_mini_batches(x, y, BATCH_SIZE)
                for batch in minibatch:
                    x_batch, y_batch = batch

                    train_feed_dict = {input_x: (x_batch - 127.5) / 128, input_y: y_batch, learning_rate: lr}
                    _, train_loss, train_acc = sess.run([train_op, loss_val, acc], train_feed_dict)
                    step += 1
                    if step % 50 == 0:
                        # c, train_loss, train_acc = sess.run([embed, loss_val, acc], feed_dict)
                        pre_loss = train_loss
                        print('%d round train loss : %f, train acc : %f' % (step, train_loss, train_acc))
                        if pre_loss * 100 < train_loss:
                            stop = True
                            break

                        pre_loss = train_loss

                        # converter = tf.lite.TFLiteConverter.from_session(sess, [input_x], [embed])
                        # tflite_model = converter.convert()
                        # with open('model_train.tflite','wb') as f:
                        #     f.write(tflite_model)

                        # tf.train.write_graph(sess.graph_def, '/home/zhangziyang/facial_recong', 'train_model.pb', as_text=False)
                        # print(c.shape)
                        # print(c)
                    if (not stop) and step % 2000 == 0:
                        path = saver.save(sess, checkpoints_dir + '/model.ckpt', global_step=step)
                        print('Save model in step:{}, path:{}'.format(step, path))
                        # converter = tf.lite.TFLiteConverter.from_session(sess, [input_x], [embed])
                        # tflite_model = converter.convert()

                        # with open('model_train.tflite','wb') as f:
                        #     f.write(tflite_model)
                        # tf.train.write_graph(sess.graph_def, 'e:/fl/tmp/cnn_data/casia/model', 'easy_model.pb', as_text=False)
                if stop:
                    break
            if stop:
                break


