import tensorflow as tf
import base_model


class Regression(base_model.Model):
    """

    """

    def __init__(self):
        super().__init__()
        self.learning_rate = 0.01
        self.graph = tf.Graph()
        self.placeholder_name = ['batch_size', 'x', 'y']
        self.var = []

    def graph_generator(self):
        """
        :return:
        """
        with self.graph.as_default():
            batch_size = tf.placeholder(tf.float32, name=self.placeholder_name[0])
            x = tf.compat.v1.placeholder(tf.float32, [None, 141], name=self.placeholder_name[1])
            y = tf.compat.v1.placeholder(tf.float32, [None, 2], name=self.placeholder_name[2])
            w = tf.Variable(tf.zeros([141, 2]))
            b = tf.Variable(tf.zeros([2]))
            y_pre = tf.nn.softmax(tf.matmul(x, w) + b)
            tf.reduce_mean(-tf.reduce_sum(y * tf.compat.v1.log(y_pre), axis=1), name="loss")
            var = self.optimizer(x, y, y_pre, w, b, batch_size)
            self.var = [w, b]
            self.assign_var(var)

    def optimizer(self, x, y, y_pre, w_init, b_init, batch_size):
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name="minimizeGradientDescent")
        :return: update w and b
        """
        delta_y = y_pre - y
        derivative_w = tf.matmul(tf.transpose(x), delta_y)
        derivative_w = derivative_w / batch_size
        derivative_b = tf.reduce_sum(delta_y, axis=0) / batch_size
        w_new = w_init - self.learning_rate * derivative_w
        b_new = b_init - self.learning_rate * derivative_b
        var = [w_new, b_new]
        return var

    def optimizer_tf(self, cost):
        # fixme: Bug still in android
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost, name="minimizeGradientDescent")


if __name__ == '__main__':
    Regression().graph_generator()

