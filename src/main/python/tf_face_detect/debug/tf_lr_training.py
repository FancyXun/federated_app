
from __future__ import print_function
import json

import tensorflow as tf
import numpy as np
from google.protobuf import json_format

mnist_x  = np.ones(shape=(1000, 10))
mnist_y  = np.ones(shape=(1000, ), dtype=np.int32)
mnist_y = np.eye(10)[mnist_y]

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
generated_graph = False

x = tf.placeholder(tf.float32, [None, 10])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([10, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

cost = tf.reshape(cost, (1,))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

if generated_graph:
    sess = tf.Session()
    sess.run(init)
    graph_def = sess.graph.as_graph_def()
    json_string = json_format.MessageToJson(graph_def)
    obj = json.loads(json_string)
    tf.compat.v1.train.write_graph(sess.graph, "./", 'lr' + '.pb', as_text=False)

    # generate txt

    trainable_var = tf.trainable_variables()
    global_var = tf.global_variables()
    with open("./" + "lr_trainable_var" + ".txt", "w") as f:
        variables_sum = 0
        for var in trainable_var:
            accumulate = 1
            for i in range(len(var.shape)):
                accumulate = var.shape[i] * accumulate
            variables_sum = accumulate + variables_sum
            f.write(var.initial_value.op.name + ";" + str(var.op.name) + "\n")
        print(variables_sum)

    with open("./" + "lr_trainable_init_var" + ".txt", "w") as f:
        variables_sum = 0
        for var in global_var:
            accumulate = 1
            for i in range(len(var.shape)):
                accumulate = var.shape[i] * accumulate
            variables_sum = accumulate + variables_sum
            if 'Momentum' not in var.initial_value.op.name:
                f.write(var.initial_value.op.name + ";" + str(var.shape) + "\n")
        print(variables_sum)

    with open("./" + "lr_feed_fetch" + ".txt", "w") as f:
        f.write(x.op.name + ";" + str(x.shape) + "\n")
        f.write(y.op.name + ";" + str(y.shape) + "\n")
        f.write(init.name + ";" + "---" + "\n")
        f.write(optimizer.name + ";" + "---" + "\n")
        f.write(cost.name + ";" + "---" + "\n")
else:
    with tf.Session() as sess:

        sess.run(init)
        for i in range(10):
            batch_xs, batch_ys = mnist_x, mnist_y
            w_var = sess.run(W)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            w_var = sess.run(W)
            # print(w_var[0][0], w_var[1][0], w_var[8][0])
            print(c[0])