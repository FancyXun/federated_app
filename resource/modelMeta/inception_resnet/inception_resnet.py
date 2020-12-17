import tensorflow as tf
import os
import numpy as np
from PIL import Image
import random
from google.protobuf import json_format
import json


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    """
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :return: The created variable
    """
    weight_decay = 0.0002
    # todo: regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay)
    #  is not available in android
    return tf.get_variable(name, shape=shape, initializer=initializer,
                           # regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay)
                           )


def bias(a, channel):
    return tf.nn.bias_add(a, tf.Variable(tf.random_normal([channel])))


def batch_normalization_layer(input_layer, dimension, base_name):
    """
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :param base_name: tensor base name
    :return: the 4D tensor after being normalized
    """
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable(base_name + '/beta', dimension, tf.float32,
                           initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(base_name + '/gamma', dimension, tf.float32,
                            initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)

    return bn_layer


def inception_resnet_stem(x):
    """

    :param x:
    :return:
    """
    out_channel = 256
    c = tf.nn.relu(bias(tf.nn.conv2d(x,
                                     create_variables(name='stem/conv0', shape=(3, 3, 3, 32)),
                                     strides=[1, 2, 2, 1], padding='VALID'), 32))

    c = tf.nn.relu(bias(tf.nn.conv2d(c,
                                     create_variables(name='stem/conv1', shape=(3, 3, 32, 32)),
                                     strides=[1, 1, 1, 1], padding='VALID'), 32))
    c = tf.nn.relu(bias(tf.nn.conv2d(c,
                                     create_variables(name='stem/conv2', shape=(3, 3, 32, 64)),
                                     strides=[1, 1, 1, 1], padding='VALID'), 64))
    c = tf.nn.max_pool(c,
                       ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    c = tf.nn.relu(bias(tf.nn.conv2d(c,
                                     create_variables(name='stem/conv3', shape=(1, 1, 64, 80)),
                                     strides=[1, 1, 1, 1], padding='SAME'), 80))
    c = tf.nn.relu(bias(tf.nn.conv2d(c,
                                     create_variables(name='stem/conv4', shape=(3, 3, 80, 192)),
                                     strides=[1, 1, 1, 1], padding='VALID'), 192))
    c = tf.nn.relu(bias(tf.nn.conv2d(c,
                                     create_variables(name='stem/conv5', shape=(3, 3, 192, out_channel)),
                                     strides=[1, 2, 2, 1], padding='SAME'), out_channel))
    b = batch_normalization_layer(c, out_channel, 'stem')
    b = tf.nn.relu(b)
    return b, out_channel


def inception_resnet_A(x, input_channel, block, scale_residual=True):
    """

    :param x:
    :param input_channel:
    :param block:
    :param scale_residual:
    :return:
    """
    out_channel = 256
    init = x
    ir1 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                       create_variables(
                                           name='resnet_A{}/conv0'.format(block).format(block),
                                           shape=(1, 1, input_channel, 32)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 32))

    ir2 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                       create_variables(
                                           name='resnet_A{}/conv1'.format(block),
                                           shape=(1, 1, input_channel, 32)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 32))

    ir2 = tf.nn.relu(bias(tf.nn.conv2d(ir2,
                                       create_variables(
                                           name='resnet_A{}/conv2'.format(block),
                                           shape=(3, 3, 32, 32)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 32))

    ir3 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                       create_variables(
                                           name='resnet_A{}/conv3'.format(block),
                                           shape=(1, 1, input_channel, 32)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 32))

    ir3 = tf.nn.relu(bias(tf.nn.conv2d(ir3,
                                       create_variables(
                                           name='resnet_A{}/conv4'.format(block),
                                           shape=(3, 3, 32, 32)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 32))

    ir3 = tf.nn.relu(bias(tf.nn.conv2d(ir3,
                                       create_variables(
                                           name='resnet_A{}/conv5'.format(block),
                                           shape=(3, 3, 32, 32)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 32))

    ir_merge = tf.concat(axis=3, values=[ir1, ir2, ir3])

    ir_conv = bias(tf.nn.conv2d(ir_merge,
                                create_variables(
                                    name='resnet_A{}/conv6'.format(block),
                                    shape=(1, 1, int(ir_merge.shape[-1]), out_channel)),
                                strides=[1, 1, 1, 1], padding='SAME'), out_channel)

    if scale_residual:
        ir_conv = ir_conv * 0.1
    out = tf.add(init, ir_conv)
    out = batch_normalization_layer(out, out_channel, 'resnet_A{}'.format(block))
    out = tf.nn.relu(out)
    return out, out_channel


def reduction_A(x, input_channel, k=192, l=224, m=256, n=384):
    r1 = tf.nn.max_pool(x,
                        ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    r2 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                      create_variables(
                                          name='reduction_A/conv0',
                                          shape=(3, 3, input_channel, n)),
                                      strides=[1, 2, 2, 1], padding='VALID'), n))

    r3 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                      create_variables(
                                          name='reduction_A/conv1',
                                          shape=(1, 1, input_channel, k)),
                                      strides=[1, 1, 1, 1], padding='SAME'), k))

    r3 = tf.nn.relu(bias(tf.nn.conv2d(r3,
                                      create_variables(
                                          name='reduction_A/conv2',
                                          shape=(3, 3, k, l)),
                                      strides=[1, 1, 1, 1], padding='SAME'), l))

    r3 = tf.nn.relu(bias(tf.nn.conv2d(r3,
                                      create_variables(
                                          name='reduction_A/conv3',
                                          shape=(3, 3, l, m)),
                                      strides=[1, 2, 2, 1], padding='VALID'), m))

    out = tf.concat(axis=3, values=[r1, r2, r3])
    out_channel = int(out.shape[-1])
    out = batch_normalization_layer(out, out_channel, "reduction_A")
    out = tf.nn.relu(out)
    return out, out_channel


def inception_resnet_B(x, input_channel, block, scale_residual=True):
    out_channel = 896
    init = x
    ir1 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                       create_variables(
                                           name='resnet_B{}/conv0'.format(block),
                                           shape=(1, 1, input_channel, 128)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 128))

    ir2 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                       create_variables(
                                           name='resnet_B{}/conv1'.format(block),
                                           shape=(1, 1, input_channel, 128)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 128))

    ir2 = tf.nn.relu(bias(tf.nn.conv2d(ir2,
                                       create_variables(
                                           name='resnet_B{}/conv2'.format(block),
                                           shape=(1, 7, 128, 128)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 128))

    ir2 = tf.nn.relu(bias(tf.nn.conv2d(ir2,
                                       create_variables(
                                           name='resnet_B{}/conv3'.format(block),
                                           shape=(7, 1, 128, 128)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 128))

    ir_merge = tf.concat(axis=3, values=[ir1, ir2])
    ir_conv = bias(tf.nn.conv2d(ir_merge,
                                create_variables(
                                    name='resnet_B{}/conv4'.format(block),
                                    shape=(1, 1, int(ir_merge.shape[-1]), out_channel)),
                                strides=[1, 1, 1, 1], padding='SAME'), out_channel)
    if scale_residual:
        ir_conv = ir_conv * 0.1

    out = tf.add(init, ir_conv)
    out = batch_normalization_layer(out, out_channel, 'resnet_B{}'.format(block))
    out = tf.nn.relu(out)
    return out, out_channel


def reduction_B(x, input_channel):
    r1 = tf.nn.max_pool(x,
                        ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    r2 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                      create_variables(
                                          name='reduction_B/conv1',
                                          shape=(1, 1, input_channel, 256)),
                                      strides=[1, 1, 1, 1], padding='SAME'), 256))

    r2 = tf.nn.relu(bias(tf.nn.conv2d(r2,
                                      create_variables(
                                          name='reduction_B/conv2',
                                          shape=(3, 3, 256, 384)),
                                      strides=[1, 2, 2, 1], padding='VALID'), 384))

    r3 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                      create_variables(
                                          name='reduction_B/conv3',
                                          shape=(1, 1, input_channel, 256)),
                                      strides=[1, 1, 1, 1], padding='SAME'), 256))

    r3 = tf.nn.relu(bias(tf.nn.conv2d(r3,
                                      create_variables(
                                          name='reduction_B/conv4',
                                          shape=(3, 3, 256, 256)),
                                      strides=[1, 2, 2, 1], padding='VALID'), 256))

    r4 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                      create_variables(
                                          name='reduction_B/conv5',
                                          shape=(1, 1, input_channel, 256)),
                                      strides=[1, 1, 1, 1], padding='SAME'), 256))

    r4 = tf.nn.relu(bias(tf.nn.conv2d(r4,
                                      create_variables(
                                          name='reduction_B/conv6',
                                          shape=(3, 3, 256, 256)),
                                      strides=[1, 1, 1, 1], padding='SAME'), 256))

    r4 = tf.nn.relu(bias(tf.nn.conv2d(r4,
                                      create_variables(
                                          name='reduction_B/conv7',
                                          shape=(3, 3, 256, 256)),
                                      strides=[1, 2, 2, 1], padding='VALID'), 256))

    out = tf.concat(axis=3, values=[r1, r2, r3, r4])
    out_channel = int(out.shape[-1])
    out = batch_normalization_layer(out, out_channel, "reduction_B")
    out = tf.nn.relu(out)
    return out, out_channel


def inception_resnet_C(x, input_channel, block, scale_residual=True):
    init = x
    out_channel = 1792
    ir1 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                       create_variables(
                                           name='resnet_C{}/conv1'.format(block),
                                           shape=(1, 1, input_channel, 128)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 128))

    ir2 = tf.nn.relu(bias(tf.nn.conv2d(x,
                                       create_variables(
                                           name='resnet_C{}/conv2'.format(block),
                                           shape=(1, 1, input_channel, 192)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 192))

    ir2 = tf.nn.relu(bias(tf.nn.conv2d(ir2,
                                       create_variables(
                                           name='resnet_C{}/conv4'.format(block),
                                           shape=(1, 3, 192, 192)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 192))

    ir2 = tf.nn.relu(bias(tf.nn.conv2d(ir2,
                                       create_variables(
                                           name='resnet_C{}/conv5'.format(block),
                                           shape=(3, 1, 192, 192)),
                                       strides=[1, 1, 1, 1], padding='SAME'), 192))

    ir_merge = tf.concat(axis=3, values=[ir1, ir2])
    ir_conv = bias(tf.nn.conv2d(ir_merge,
                                create_variables(
                                    name='resnet_C{}/conv6'.format(block),
                                    shape=(1, 1, int(ir_merge.shape[-1]), out_channel)),
                                strides=[1, 1, 1, 1], padding='SAME'), out_channel)

    if scale_residual:
        ir_conv = ir_conv * 0.1

    out = tf.add(init, ir_conv)
    out = batch_normalization_layer(out, out_channel, 'resnet_C{}/conv7'.format(block))
    out = tf.nn.relu(out)
    return out, out_channel


def center_loss(x, y, alpha=0.5, nb_classes=10, embed_dim=128):
    centers = tf.Variable(tf.random.uniform(shape=(nb_classes, embed_dim)), name='centers', trainable=False)
    delta_centers = tf.matmul(tf.transpose(y), tf.matmul(y, centers) - x)
    center_counts = tf.reshape(tf.reduce_sum(tf.transpose(y), axis=1) + 1, (-1, 1))
    delta_centers /= center_counts
    new_centers = tf.subtract(centers, alpha * delta_centers)
    centers.assign(new_centers)
    result = tf.reshape(tf.reduce_sum(tf.square(x - tf.matmul(y, centers)), axis=1), (-1, 1))
    return result


class InceptionResNet(object):

    def __init__(self, nb_classes=10, learning_rate=0.001, scale=True, height=299, width=299):
        self.nb_classes = nb_classes
        self.scale = scale
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3], name="x")
        self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, nb_classes], name="y")
        self.learning_rate = learning_rate

    def build(self):

        x, out_channel = inception_resnet_stem(self.input_x)

        # 5 x Inception Resnet A
        for i in range(5):
            x, out_channel = inception_resnet_A(x, out_channel, i, scale_residual=self.scale)
        # Reduction A - From Inception v4
        x, out_channel = reduction_A(x, out_channel, k=192, l=192, m=256, n=384)

        # 10 x Inception Resnet B
        for i in range(10):
            x, out_channel = inception_resnet_B(x, out_channel, i, scale_residual=self.scale)

        # Reduction Resnet B
        x, out_channel = reduction_B(x, out_channel)

        # 5 x Inception Resnet C
        for i in range(5):
            x, out_channel = inception_resnet_C(x, out_channel, i, scale_residual=self.scale)

        # Average Pooling
        x = tf.nn.max_pool(x,
                           ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')

        x = tf.nn.dropout(x, rate=0.2)

        x = tf.layers.flatten(x)
        out = tf.layers.dense(x, self.nb_classes)
        side_out = center_loss(x, self.input_label, alpha=0.5, nb_classes=self.nb_classes, embed_dim=1792)
        categorical_crossentropy = tf.nn.softmax_cross_entropy_with_logits_v2(self.input_label, out)
        zero_loss = 0.5 * tf.reduce_sum(side_out)
        loss = categorical_crossentropy + 0.1 * zero_loss
        optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(loss, name="minimizeGradientDescent")

        return out, side_out, optimizer, loss

    @staticmethod
    def loss(logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


def load_data(path):
    image_cate = os.listdir(path)
    for p in image_cate:
        if p.endswith('txt'):
            image_cate.remove(p)

    cate_num = []
    for i in range(len(image_cate)):
        cate_num.append(len(os.listdir(os.path.join(path, image_cate[i]))))
    size = 2000
    x = np.zeros(shape=(size, 182, 182, 3), dtype=np.float32)
    y = np.zeros(shape=(size,), dtype=np.int32)
    f = 0
    s = 0
    for i in range(len(image_cate)):

        cate_dir = os.listdir(os.path.join(path, image_cate[i]))
        for img in cate_dir:
            if s >= 2000:
                break
            s += 1
            im = Image.open(os.path.join(path, image_cate[i] + '/' + img))
            x[f] = np.array(im)
            y[f] = int(i)
            f += 1
    rl1 = list(range(x.shape[0]))
    random.shuffle(rl1)
    x = x[rl1]
    y = y[rl1]
    y = np.eye(len(cate_num))[y]
    rl = random.sample(list(range(x.shape[0])), 1000)
    x_eval = x[rl]
    y_eval = y[rl]
    return x, y, x_eval, y_eval, len(cate_num)


model = InceptionResNet(nb_classes=1006, scale=True, height=182, width=182)
out, side_out, optimizer, loss = model.build()
print(out)
print(side_out)
training_epochs = 5
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
graph_def = sess.graph.as_graph_def()
json_string = json_format.MessageToJson(graph_def)
obj = json.loads(json_string)
tf.compat.v1.train.write_graph(sess.graph, "./", "inception_resnet.pb", as_text=False)
print("save trainable variable...")
trainable_var = tf.trainable_variables()
with open("inception_resnet_trainable_var.txt", "a+") as f:
    variables_sum = 0
    for var in trainable_var:
        accumulate = 1
        for i in range(len(var.shape)):
            accumulate = var.shape[i] * accumulate
        variables_sum = accumulate + variables_sum
        f.write(var.op.name + ":" + str(var.shape) + "\n")
    print(variables_sum)

with open("inception_resnet_trainable_init_var.txt", "a+") as f:
    variables_sum = 0
    for var in trainable_var:
        accumulate = 1
        for i in range(len(var.shape)):
            accumulate = var.shape[i] * accumulate
        variables_sum = accumulate + variables_sum
        f.write(var.initial_value.op.name + ":" + str(var.shape) + "\n")
    print(variables_sum)

with open("inception_resnet_feed_fetch.txt", "a+") as f:
    f.write(model.input_label.op.name + ":" + str(model.input_label.shape) + "\n")
    f.write(model.input_x.op.name + ":" + str(model.input_x.shape) + "\n")
    f.write("---------------------------------------------------------"+"\n")
    f.write(init.name  + ":" + "---" + "\n")
    f.write("---------------------------------------------------------"+"\n")
    f.write(optimizer.name + ":" + "---" + "\n")
    f.write("---------------------------------------------------------"+"\n")
    f.write(loss.name + ":" + "---" + "\n")

