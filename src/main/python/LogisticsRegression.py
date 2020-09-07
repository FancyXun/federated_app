import tensorflow as tf
import numpy as np
graph = tf.Graph()


def model():
    with graph.as_default():
        x = tf.compat.v1.placeholder(tf.float32, [None, None], name='x')
        y = tf.compat.v1.placeholder(tf.float32, [None, None], name='y')
        w = tf.Variable(tf.zeros([10, 10]), validate_shape=False, name='w')
        b = tf.Variable(tf.zeros([10]), validate_shape=False,name='b')
        y_pre = tf.nn.softmax(tf.matmul(x, w) + b)
        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.log(y_pre), axis=1), name="cost")
        learning_rate = 0.01
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost, name="minimizeGradientDescent")

        init = tf.compat.v1.global_variables_initializer()
        training_epochs = 50
        sess = tf.compat.v1.Session()
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            for i in range(10):
                batch_xs = np.random.uniform(0, 1, (10, 10))
                batch_ys = np.random.uniform(0, 1, (10, 10))
                feeds_train = {x: batch_xs, y: batch_ys}
                sess.run(optimizer, feed_dict=feeds_train)
                avg_cost += sess.run(cost, feed_dict=feeds_train) / 10
        print("Done")
        return cost, optimizer, x, y


cost, optimizer, x, y = model()
tf.compat.v1.train.write_graph(graph, "./", 'graph.pb', as_text=False)