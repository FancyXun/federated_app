import tensorflow as tf

res1_3 = [
    {'filters': 64, 'kernel_size': 3, 'strides': 2, 'w_init': 'xavier', 'padding': 'same', 'suffix': '1'},
    {'filters': 64, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '2'},
    {'filters': 64, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '3'},
]

res2_3 = [
    {'filters': 128, 'kernel_size': 3, 'strides': 2, 'w_init': 'xavier', 'padding': 'same', 'suffix': '1'},
    {'filters': 128, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '2'},
    {'filters': 128, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '3'},
]

res2_5 = [
    {'filters': 128, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '4'},
    {'filters': 128, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '5'},
]

res3_3 = [
    {'filters': 256, 'kernel_size': 3, 'strides': 2, 'w_init': 'xavier', 'padding': 'same', 'suffix': '1'},
    {'filters': 256, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '2'},
    {'filters': 256, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '3'},
]

res3_5 = [
    {'filters': 256, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '4'},
    {'filters': 256, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '5'},
]

res3_7 = [
    {'filters': 256, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '6'},
    {'filters': 256, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '7'},
]

res3_9 = [
    {'filters': 256, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '8'},
    {'filters': 256, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '9'},
]

res4_3 = [
    {'filters': 512, 'kernel_size': 3, 'strides': 2, 'w_init': 'xavier', 'padding': 'same', 'suffix': '1'},
    {'filters': 512, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '2'},
    {'filters': 512, 'kernel_size': 3, 'strides': 1, 'w_init': 'gaussian', 'padding': 'same', 'suffix': '3'},
]


class Model(object):

    def __init__(self, inputs):

        self.embeddings = self.network(inputs)

    @staticmethod
    def network(inputs, embedding_dim=512):

        def prelu(inputs, name=''):
            alpha = tf.get_variable(name, shape=inputs.get_shape()[-1],
                                    initializer=tf.constant_initializer(0.0), dtype=inputs.dtype)
            return tf.maximum(alpha * inputs, inputs)

        def conv(inputs, filters, kernel_size, strides, w_init, padding='same', suffix='', scope=None):
            conv_name = 'conv' + suffix
            relu_name = 'relu' + suffix

            with tf.name_scope(name=scope):
                if w_init == 'xavier':   w_init = tf.contrib.layers.xavier_initializer(uniform=True)
                if w_init == 'gaussian': w_init = tf.contrib.layers.xavier_initializer(uniform=False)
                input_shape = inputs.get_shape().as_list()
                net = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding=padding,
                                       kernel_initializer=w_init, name=conv_name)
                output_shape = net.get_shape().as_list()
                print("=================================================================================")
                print("layer:%8s    input shape:%8s   output shape:%8s" % (
                conv_name, str(input_shape), str(output_shape)))
                print("---------------------------------------------------------------------------------")
                net = prelu(net, name=relu_name)
                return net

        def resnet_block(net, blocks, suffix=''):
            n = len(blocks)
            for i in range(n):
                if n == 2 and i == 0: identity = net
                net = conv(inputs=net,
                           filters=blocks[i]['filters'],
                           kernel_size=blocks[i]['kernel_size'],
                           strides=blocks[i]['strides'],
                           w_init=blocks[i]['w_init'],
                           padding=blocks[i]['padding'],
                           suffix=suffix + '_' + blocks[i]['suffix'],
                           scope='conv' + suffix + '_' + blocks[i]['suffix'])

                if n == 3 and i == 0: identity = net
            return identity + net

        net = inputs
        for suffix, blocks in zip(('1', '2', '2', '3', '3', '3', '3', '4'),
                                  (res1_3, res2_3, res2_5, res3_3, res3_5, res3_7, res3_9, res4_3)):
            net = resnet_block(net, blocks, suffix=suffix)

        net = tf.layers.flatten(net)
        embeddings = tf.layers.dense(net, units=embedding_dim,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        return embeddings

