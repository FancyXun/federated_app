import tensorflow as tf
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-p', '--path', help='the graph root path ')
parse.add_argument('-g_name', '--graph_name', help='the graph name')

args = vars(parse.parse_args())
path = args["path"]
graph_name = args["graph_name"]
graph = tf.Graph()

with graph.as_default():
    x = tf.compat.v1.placeholder(tf.float32, [None, None], name='x')
    y = tf.compat.v1.placeholder(tf.float32, [None, None], name='y')
    w = tf.Variable(tf.zeros([2, 2], name="w/init"), validate_shape=False, name='w')
    b = tf.Variable(tf.zeros([2], name="b/init"), validate_shape=False, name='b')
    y_pre = tf.nn.softmax(tf.matmul(x, w) + b, name="y_pre")
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.compat.v1.log(y_pre), axis=1), name="cost")
    learning_rate = 0.01
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost,
                                                                                    name="minimizeGradientDescent")
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)

tf.compat.v1.train.write_graph(graph, path, graph_name, as_text=False)