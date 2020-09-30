import tensorflow as tf
import base_model
import numpy as np

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
            batch_size = tf.compat.v1.placeholder(tf.float32, name=self.placeholder_name[0])
            x = tf.compat.v1.placeholder(tf.float32, [None, 141], name=self.placeholder_name[1])
            y = tf.compat.v1.placeholder(tf.float32, [None, 2], name=self.placeholder_name[2])

            #auc_test
            labels = tf.placeholder(shape=(None,), dtype=tf.float32,name='labels')
            predictions = tf.placeholder(shape=(None,), dtype=tf.float32,name='predictions')
            auc = self.auc_tf(labels,predictions)
            local_var = tf.local_variables()
            init_new_vars_op = tf.initialize_variables(local_var)

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

    def auc_tf(self,lab,pre):
        auc_value, auc_op = tf.metrics.auc(lab, pre, num_thresholds=1000,name='auc_pair')
        return auc_op

if __name__ == '__main__':
    Regression().graph_generator()
    print('project_ready!')



""" java code for calculating auc with a given labels_predictions pair

        float labarr [] = {0,0,0,0,0,0,1,1,1,1,1,1};
        float prearr [] = {0.3f,0.4f,0.6f,0.2f,0.7f,0.5f,0.1f,0.9f,0.7f,0.5f,0.9f,0.8f};
        float init [] = new float [1000];
        System.out.println(init);
        Tensor x = Tensor.create(labarr);
        Tensor y = Tensor.create(prearr);
        System.out.println("tensor,ok! :"+x+y);
        System.out.println("array,ok!" + labarr + prearr);

        s.runner().feed("auc_pair/true_positives/Initializer/zeros",Tensor.create(init)).addTarget("auc_pair/true_positives/Assign").run();
        s.runner().feed("auc_pair/false_positives/Initializer/zeros",Tensor.create(init)).addTarget("auc_pair/false_positives/Assign").run();
        s.runner().feed("auc_pair/true_negatives/Initializer/zeros",Tensor.create(init)).addTarget("auc_pair/true_negatives/Assign").run();
        s.runner().feed("auc_pair/false_negatives/Initializer/zeros",Tensor.create(init)).addTarget("auc_pair/false_negatives/Assign").run();
        Tensor z = s.runner().feed("labels", x).feed("predictions",y).fetch("auc_pair/update_op").run().get(0);

        System.out.println("AUC: "+z.floatValue());
        System.out.println("to the end");
"""