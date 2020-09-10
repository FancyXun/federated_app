import tensorflow as tf
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-p', '--path', help='the graph root path ')
parse.add_argument('-g_name', '--graph_name', help='the graph name')

args = vars(parse.parse_args())
path = args["path"]
graph_name = args["graph_name"]
graph = tf.Graph()


def optimizer(x, y, y_pre, w_init, b_init, learning_rate, batch_size):
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,
                                                                          name="minimizeGradientDescent")
    :return: update w and b
    """
    delta_y = y_pre - y
    derivative_w = tf.matmul(tf.transpose(x), delta_y)
    derivative_w = derivative_w / batch_size
    derivative_b = tf.reduce_sum(delta_y, axis=0) / batch_size
    w_new = w_init - learning_rate * derivative_w
    b_new = b_init - learning_rate * derivative_b
    return w_new, b_new


with graph.as_default():
    batch_size = tf.placeholder(tf.float32, name='batch_size')
    x = tf.compat.v1.placeholder(tf.float32, [None, 141], name='x')
    y = tf.compat.v1.placeholder(tf.float32, [None, 2], name='y')
    w = tf.Variable(tf.zeros([141, 2], name="w/init"), validate_shape=False, name='w')
    b = tf.Variable(tf.zeros([2], name="b/init"), validate_shape=False, name='b')
    y_pre = tf.nn.softmax(tf.matmul(x, w) + b, name="y_pre")
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.compat.v1.log(y_pre), axis=1), name="cost")
    learning_rate = 0.01
    w_update, b_update = optimizer(x, y, y_pre, w, b, learning_rate, batch_size)
    w_assign = w.assign(w_update, name="w_assign")
    b_assign = b.assign(b_update, name="b_assign")
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)

tf.compat.v1.train.write_graph(graph, path, graph_name, as_text=False)