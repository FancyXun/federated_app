import tensorflow as tf
import numpy as np
import os
import json
from google.protobuf import json_format


def prelu(input, name):
    alphas = tf.get_variable(name + '_alphas', input.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(input)
    neg = alphas * tf.nn.relu(-input)
    return tf.add(pos, neg, name=name)


def weight_variable(shape, stddev=0.2, name=None):
    # initial = tf.truncated_normal(shape, stddev=stddev)
    initial = tf.glorot_uniform_initializer()
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, shape=shape, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def ce_loss(logit, label, reg_ratio=0.):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=label))
    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label))
    # reg_losses = tf.add_n(tf.get_collection('losses'))
    # return cross_entropy_loss + reg_ratio * reg_losses
    return cross_entropy_loss


def agular_margin_softmax_loss(embedding, label, step, margin=4):
    # batch_num = int(batch_num.name.split(':')[-1])
    it = step
    embeddig_norm = tf.norm(embedding, axis=1)
    embed_dim = embedding.get_shape()[1]
    initial = tf.glorot_uniform_initializer()
    weights = tf.get_variable(name='softmax_weights', shape=[embed_dim, 10575],
                              initializer=initial)
    weights_norm = tf.nn.l2_normalize(weights, axis=0)
    stad_logits = tf.matmul(embedding, weights_norm)
    # batch_size = tf.shape(embedding)[0]
    batch_size = 64
    spr_label = label
    sample_2d_label_idx = tf.stack([tf.constant(list(range(batch_size)), tf.int32), spr_label], axis=1)
    sample_logits = tf.gather_nd(stad_logits, sample_2d_label_idx)

    cos_theta = tf.div(sample_logits, embeddig_norm)
    cos_2_power = tf.pow(cos_theta, 2)
    cos_4_power = tf.pow(cos_theta, 4)
    sign_cos_theta = tf.sign(cos_theta)
    neg_one_power_k = tf.multiply(tf.sign(2 * cos_2_power - 1), sign_cos_theta)
    minus_double_k = 2 * sign_cos_theta + neg_one_power_k - 3
    phi_theta = neg_one_power_k * (8 * cos_4_power - 8 * cos_2_power) + minus_double_k

    margin_logits = tf.multiply(phi_theta, embeddig_norm)
    combined_logits = tf.add(stad_logits, tf.scatter_nd(sample_2d_label_idx,
                                                        tf.subtract(margin_logits, sample_logits),
                                                        (batch_size, 10575)))

    lamb = tf.maximum(5., 1500. / (1 + 1.0 * it))
    f = 1.0 / (1.0 + lamb)
    ff = 1.0 - f
    final_logits = ff * stad_logits + f * combined_logits
    a_softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=final_logits))
    return a_softmax_loss, final_logits, stad_logits


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


class Sphere:

    def __init__(self, nb_classes=10575, learning_rate=0.001, scale=True, height=112, width=96):
        self.nb_classes = nb_classes
        self.scale = scale
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3], name="x")
        self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_classes], name="y")
        self.learning_rate = learning_rate

    def build(self):
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
        # input_image = tf.image.resize_images(images=self.input_x, size=(112, 96))
        PAD = [[0, 0], [1, 0], [1, 0], [0, 0]]
        # input_x = tf.placeholder(tf.float32, [None] + [112, 96] + [3], name='input_x')
        # input_y = tf.placeholder(tf.int32, [None, nb_classes], name='input_y')

        # pad11 = tf.pad(self.input_x, PAD, "CONSTANT")
        conv11 = tf.nn.conv2d(self.input_x, weights['c1_1'], strides=[1, 2, 2, 1], padding=PAD, name='c_11')
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
        # pad21 = tf.pad(res_1, PAD, "CONSTANT")
        conv21 = tf.nn.conv2d(res_1, weights['c2_1'], strides=[1, 2, 2, 1], padding=PAD, name='c_21')
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
        # pad31 = tf.pad(res_22, PAD, "CONSTANT")
        conv31 = tf.nn.conv2d(res_22, weights['c3_1'], strides=[1, 2, 2, 1], padding=PAD, name='c_31')
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
        # pad41 = tf.pad(res_34, PAD, "CONSTANT")
        conv41 = tf.nn.conv2d(res_34, weights['c4_1'], strides=[1, 2, 2, 1], padding=PAD, name='c_41')
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
        # fc_1 = tf.layers.dense(flat1, 512, name='fc_1')
        # fc_1 = tf.matmul(flat1, weights['fc5'])+ weights['b5']
        fc_1 = tf.nn.bias_add(tf.matmul(flat1, weights['fc5']), weights['b5'],
                              name='feature_embed')
        # h_fc_1 = tf.nn.bias_add(fc_1, weights['b5'])
        logits = tf.layers.dense(fc_1, self.nb_classes, name='pred_logits')
        loss_val = ce_loss(logits, self.input_label)

        train_op = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).minimize(loss_val)
        return fc_1, loss_val, train_op


tf.reset_default_graph()
model = Sphere(nb_classes=10575, scale=True, height=112, width=96)
embedding, loss, optimizer = model.build()
print(embedding)
training_epochs = 5
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
graph_def = sess.graph.as_graph_def()
json_string = json_format.MessageToJson(graph_def)
obj = json.loads(json_string)
tf.compat.v1.train.write_graph(sess.graph, "./", "sphere2.pb", as_text=False)
print("save trainable variable...")
trainable_var = tf.trainable_variables()
with open("sphere2_trainable_var.txt", "a+") as f:
    variables_sum = 0
    for var in trainable_var:
        accumulate = 1
        for i in range(len(var.shape)):
            accumulate = var.shape[i] * accumulate
        variables_sum = accumulate + variables_sum
        f.write(var.op.name + ":" + str(var.shape) + "\n")
    print(variables_sum)

with open("sphere2_trainable_init_var.txt", "a+") as f:
    variables_sum = 0
    for var in trainable_var:
        accumulate = 1
        for i in range(len(var.shape)):
            accumulate = var.shape[i] * accumulate
        variables_sum = accumulate + variables_sum
        f.write(var.initial_value.op.name + ":" + str(var.shape) + "\n")
    print(variables_sum)

with open("sphere2_feed_fetch.txt", "a+") as f:
    f.write(model.input_label.op.name + ":" + str(model.input_label.shape) + "\n")
    f.write(model.input_x.op.name + ":" + str(model.input_x.shape) + "\n")
    f.write("---------------------------------------------------------" + "\n")
    f.write(init.name + ":" + "---" + "\n")
    f.write("---------------------------------------------------------" + "\n")
    f.write(optimizer.name + ":" + "---" + "\n")
    f.write("---------------------------------------------------------" + "\n")
    f.write(loss.name + ":" + "---" + "\n")