import tensorflow as tf
import numpy as np
# input image with 10x10 shape for 3 channels
# filter with 10x10 shape for each input channel

N_in_channel = 3
N_out_channel_mul = 8
x = tf.random_normal([1, 10, 10, N_in_channel])
f = tf.random_normal([10, 10, N_in_channel, N_out_channel_mul])
y = tf.nn.depthwise_conv2d(x, f, strides=[1, 1, 1, 1], padding="VALID", data_format="NHWC")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_data, f_data, y_conv = sess.run([x, f, y])

y_s = np.squeeze(y_conv)
for i in range(N_in_channel):
    for j in range(N_out_channel_mul):
        print("np: %f, tf:%f" % (np.sum(x_data[0, :, :, i] * f_data[:, :, i, j]), y_s[i * N_out_channel_mul + j]))