import tensorflow as tf


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    """
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :return: The created variable
    """
    weight_decay = 0.0002
    return tf.get_variable(name, shape=shape, initializer=initializer,
                           regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))


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
    out_channel = 256
    c = tf.nn.conv2d(x,
                     create_variables(name='stem/conv0', shape=(3, 3, 3, 32)),
                     strides=[1, 2, 2, 1], padding='VALID')
    c = tf.nn.relu(c)
    c = tf.nn.conv2d(c,
                     create_variables(name='stem/conv1', shape=(3, 3, 32, 32)),
                     strides=[1, 1, 1, 1], padding='VALID')
    c = tf.nn.relu(c)
    c = tf.nn.conv2d(c,
                     create_variables(name='stem/conv2', shape=(3, 3, 32, 64)),
                     strides=[1, 1, 1, 1], padding='VALID')
    c = tf.nn.relu(c)
    c = tf.nn.max_pool(c,
                       ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    c = tf.nn.conv2d(c,
                     create_variables(name='stem/conv3', shape=(1, 1, 64, 80)),
                     strides=[1, 1, 1, 1], padding='SAME')
    c = tf.nn.relu(c)
    c = tf.nn.conv2d(c,
                     create_variables(name='stem/conv4', shape=(3, 3, 80, 192)),
                     strides=[1, 1, 1, 1], padding='VALID')
    c = tf.nn.relu(c)
    c = tf.nn.conv2d(c,
                     create_variables(name='stem/conv5', shape=(3, 3, 192, out_channel)),
                     strides=[1, 2, 2, 1], padding='SAME')
    c = tf.nn.relu(c)
    b = batch_normalization_layer(c, out_channel, 'stem')
    b = tf.nn.relu(b)
    return b, out_channel


def inception_resnet_A(x, input_channel, block, scale_residual=True):
    out_channel = 256
    init = x
    ir1 = tf.nn.conv2d(x,
                       create_variables(name='resnet_A{}/conv0'.format(block).format(block), shape=(1, 1, input_channel, 32)),
                       strides=[1, 1, 1, 1], padding='SAME')
    ir1 = tf.nn.relu(ir1)

    ir2 = tf.nn.conv2d(x,
                       create_variables(name='resnet_A{}/conv1'.format(block), shape=(1, 1, input_channel, 32)),
                       strides=[1, 1, 1, 1], padding='SAME')
    ir2 = tf.nn.relu(ir2)

    ir2 = tf.nn.conv2d(ir2,
                       create_variables(name='resnet_A{}/conv2'.format(block), shape=(3, 3, 32, 32)),
                       strides=[1, 1, 1, 1], padding='SAME')
    ir2 = tf.nn.relu(ir2)

    ir3 = tf.nn.conv2d(x,
                       create_variables(name='resnet_A{}/conv3'.format(block), shape=(1, 1, input_channel, 32)),
                       strides=[1, 1, 1, 1], padding='SAME')
    ir3 = tf.nn.relu(ir3)

    ir3 = tf.nn.conv2d(ir3,
                       create_variables(name='resnet_A{}/conv4'.format(block), shape=(3, 3, 32, 32)),
                       strides=[1, 1, 1, 1], padding='SAME')
    ir3 = tf.nn.relu(ir3)

    ir3 = tf.nn.conv2d(ir3,
                       create_variables(name='resnet_A{}/conv5'.format(block), shape=(3, 3, 32, 32)),
                       strides=[1, 1, 1, 1], padding='SAME')
    ir3 = tf.nn.relu(ir3)

    ir_merge = tf.concat(axis=3, values=[ir1, ir2, ir3])

    ir_conv = tf.nn.conv2d(ir_merge,
                           create_variables(name='resnet_A{}/conv6'.format(block), shape=(1, 1, int(ir_merge.shape[-1]), out_channel)),
                           strides=[1, 1, 1, 1], padding='SAME')

    if scale_residual:
        ir_conv = ir_conv * 0.1
    out = tf.add(init, ir_conv)
    out = batch_normalization_layer(out, out_channel, 'resnet_A{}'.format(block))
    out = tf.nn.relu(out)
    return out, out_channel


def reduction_A(x, input_channel, k=192, l=224, m=256, n=384):

    r1 = tf.nn.max_pool(x,
                        ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    r2 = tf.nn.conv2d(x,
                      create_variables(name='reduction_A/conv0', shape=(3, 3, input_channel, n)),
                      strides=[1, 2, 2, 1], padding='VALID')
    r2 = tf.nn.relu(r2)

    r3 = tf.nn.conv2d(x,
                      create_variables(name='reduction_A/conv1', shape=(1, 1, input_channel, k)),
                      strides=[1, 1, 1, 1], padding='SAME')
    r3 = tf.nn.relu(r3)

    r3 = tf.nn.conv2d(r3,
                      create_variables(name='reduction_A/conv2', shape=(3, 3, k, l)),
                      strides=[1, 1, 1, 1], padding='SAME')
    r3 = tf.nn.relu(r3)

    r3 = tf.nn.conv2d(r3,
                      create_variables(name='reduction_A/conv3', shape=(3, 3, l, m)),
                      strides=[1, 2, 2, 1], padding='VALID')
    r3 = tf.nn.relu(r3)

    out = tf.concat(axis=3, values=[r1, r2, r3])
    out = batch_normalization_layer(out, int(out.shape[-1]), "reduction_A")
    out = tf.nn.relu(out)
    return out


def inception_resnet_B(x, scale_residual=True):
    out_channel = 896
    init = x
    ir1 = tf.nn.conv2d(x,
                       create_variables(name='conv0', shape=(1, 1, 256, 128)),
                       strides=[1, 1, 1, 1], padding='SAME')

    ir2 = tf.nn.conv2d(x,
                       create_variables(name='conv0', shape=(1, 1, 256, 128)),
                       strides=[1, 1, 1, 1], padding='SAME')
    ir2 = tf.nn.conv2d(ir2,
                       create_variables(name='conv0', shape=(1, 7, 128, 128)),
                       strides=[1, 1, 1, 1], padding='SAME')

    ir2 = tf.nn.conv2d(ir2,
                       create_variables(name='conv0', shape=(7, 1, 128, 128)),
                       strides=[1, 1, 1, 1], padding='SAME')

    ir_merge = tf.concat(axis=3, values=[ir1, ir2])
    ir_conv = tf.nn.conv2d(ir_merge,
                           create_variables(name='conv0', shape=(3, 3, 128, out_channel)),
                           strides=[1, 1, 1, 1], padding='SAME')
    if scale_residual:
        ir_conv = ir_conv * 0.1

    out = tf.add(init, ir_conv)
    out = batch_normalization_layer(out, out_channel)
    out = tf.nn.relu(out)
    return out


def reduction_resnet_B(x):
    input_channel = 0
    out_channel = 256
    r1 = tf.nn.max_pool(x,
                        ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    r2 = tf.nn.conv2d(x,
                      create_variables(name='conv1', shape=(1, 1, input_channel, 256)),
                      strides=[1, 1, 1, 1], padding='SAME')
    r2 = tf.nn.conv2d(r2,
                      create_variables(name='conv1', shape=(3, 3, 256, 384)),
                      strides=[1, 1, 1, 1], padding='VALID')
    r3 = tf.nn.conv2d(x,
                      create_variables(name='conv1', shape=(1, 1, input_channel, 256)),
                      strides=[1, 1, 1, 1], padding='SAME')
    r3 = tf.nn.conv2d(r3,
                      create_variables(name='conv1', shape=(3, 3, 256, 256)),
                      strides=[1, 1, 1, 1], padding='VALID')
    r4 = tf.nn.conv2d(x,
                      create_variables(name='conv1', shape=(1, 1, input_channel, 256)),
                      strides=[1, 1, 1, 1], padding='SAME')
    r4 = tf.nn.conv2d(r4,
                      create_variables(name='conv1', shape=(3, 3, 256, 256)),
                      strides=[1, 1, 1, 1], padding='VALID')
    r4 = tf.nn.conv2d(r4,
                      create_variables(name='conv1', shape=(3, 3, 256, 256)),
                      strides=[1, 1, 1, 1], padding='VALID')

    out = tf.concat(axis=3, values=[r1, r2, r3, r4])
    out = batch_normalization_layer(out, out_channel)
    out = tf.nn.relu(out)
    return out


def inception_resnet_C(x, scale_residual=True):
    init = x
    input_channel = 0
    out_channel = 1792
    ir1 = tf.nn.conv2d(x,
                       create_variables(name='conv0', shape=(1, 1, input_channel, 128)),
                       strides=[1, 1, 1, 1], padding='SAME')

    ir2 = tf.nn.conv2d(x,
                       create_variables(name='conv0', shape=(1, 1, 256, 192)),
                       strides=[1, 1, 1, 1], padding='SAME')
    ir2 = tf.nn.conv2d(ir2,
                       create_variables(name='conv0', shape=(1, 3, 128, 192)),
                       strides=[1, 1, 1, 1], padding='SAME')

    ir2 = tf.nn.conv2d(ir2,
                       create_variables(name='conv0', shape=(3, 1, 128, 192)),
                       strides=[1, 1, 1, 1], padding='SAME')

    ir_merge = tf.concat(axis=3, values=[ir1, ir2])
    ir_conv = tf.nn.conv2d(ir_merge,
                           create_variables(name='conv0', shape=(3, 3, 192, out_channel)),
                           strides=[1, 1, 1, 1], padding='SAME')

    if scale_residual:
        ir_conv = ir_conv * 0.1

    out = tf.add(init, ir_conv)
    out = batch_normalization_layer(out, out_channel)
    out = tf.nn.relu(out)
    return out


class InceptionResNet(object):

    def __init__(self, scale=True):
        self.placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
        self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        x, out_channel = inception_resnet_stem(self.placeholder)

        # 5 x Inception Resnet A
        for i in range(5):
            x, out_channel = inception_resnet_A(x, out_channel, i, scale_residual=scale)
        # Reduction A - From Inception v4
        x = reduction_A(x, out_channel, k=192, l=192, m=256, n=384)

        # 10 x Inception Resnet B
        for i in range(10):
            x = inception_resnet_B(x, scale_residual=scale)

        # Reduction Resnet B
        x = reduction_resnet_B(x)

        # 5 x Inception Resnet C
        for i in range(5):
            x = inception_resnet_C(x, scale_residual=scale)

        # Average Pooling
        x = tf.nn.max_pool(x,
                           ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.nn.dropout(x, rate=0.2)
        # Dropout


InceptionResNet()
