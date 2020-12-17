import tensorflow as tf
import numpy as np
import os 
import json
from google.protobuf import json_format

def prelu(input, name,trainable=True):
    alphas = tf.get_variable(name + '_alphas', input.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32,trainable=trainable)
    pos = tf.nn.relu(input)
    neg = alphas * tf.nn.relu(-input)
    return tf.add(pos, neg, name=name)

def weight_variable(shape, stddev=0.2, name=None,trainable=True):
    # initial = tf.truncated_normal(shape, stddev=stddev)
    initial = tf.glorot_uniform_initializer()
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, shape=shape, initializer=initial, trainable = trainable)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, shape=shape, initializer=initial)

def ce_loss( logit, label, reg_ratio=0.):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=label))
    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label))
    # reg_losses = tf.add_n(tf.get_collection('losses'))
    # return cross_entropy_loss + reg_ratio * reg_losses
    return cross_entropy_loss

def agular_margin_softmax_loss(embedding, label, step, margin=4):
    # batch_num = int(batch_num.name.split(':')[-1])
    it = step
    embeddig_norm = tf.norm(embedding, axis=1)
    embed_dim = embedding.get_shape()[1]
    initial = tf.glorot_uniform_initializer()
    weights = tf.get_variable(name='softmax_weights', shape=[embed_dim, 10575],
                              initializer=initial)
    weights_norm = tf.nn.l2_normalize(weights, axis=0)
    stad_logits = tf.matmul(embedding, weights_norm)
    # batch_size = tf.shape(embedding)[0]
    batch_size = 64
    spr_label = label
    sample_2d_label_idx = tf.stack([tf.constant(list(range(batch_size)), tf.int32), spr_label], axis=1)
    sample_logits = tf.gather_nd(stad_logits, sample_2d_label_idx)

    cos_theta = tf.div(sample_logits, embeddig_norm)
    cos_2_power = tf.pow(cos_theta, 2)
    cos_4_power = tf.pow(cos_theta, 4)
    sign_cos_theta = tf.sign(cos_theta)
    neg_one_power_k = tf.multiply(tf.sign(2*cos_2_power-1), sign_cos_theta)
    minus_double_k = 2*sign_cos_theta + neg_one_power_k - 3
    phi_theta = neg_one_power_k * (8*cos_4_power - 8*cos_2_power) + minus_double_k

    margin_logits = tf.multiply(phi_theta, embeddig_norm)
    combined_logits = tf.add(stad_logits, tf.scatter_nd(sample_2d_label_idx,
                                                        tf.subtract(margin_logits, sample_logits),
                                                        (batch_size, 10575)))

    lamb = tf.maximum(5., 1500./(1+1.0*it))
    f = 1.0/(1.0+lamb)
    ff = 1.0 - f
    final_logits = ff*stad_logits + f*combined_logits
    a_softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=final_logits))
    return a_softmax_loss, final_logits, stad_logits

def center_loss(features, label, alfa, nb_classes):
    # label: (N, nb_classes), features: (N, embed_dim), centers: (nb_classes, embed_dim)
    embed_dim = features.get_shape()[1]
    centers = tf.get_variable('centers', [nb_classes, embed_dim], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # label_n = np.cast(np.array(label), dtype=np.int32)


    label = tf.cast(label, tf.float32)
    # label = np.eye(nb_classes, dtype=np.float32)[label]

    diff = tf.matmul(label, tf.matmul(label, centers)-features, transpose_a=True)
    center_count = tf.reduce_sum(tf.transpose(label), axis=1, keepdims=True) + 1
    diff = diff / center_count
    centers = centers - alfa * diff
    with tf.control_dependencies([centers]):
        res = tf.reduce_sum(features - tf.matmul(label, centers), axis=1)
        loss = tf.reduce_mean(tf.square(res))
    return loss, centers

class Sphere:
    
    def __init__(self, nb_classes=10575, learning_rate=0.001, scale=True, height=112, width=96):
        self.nb_classes = nb_classes
        self.scale = scale
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3], name="x")
        self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_classes], name="y")
        self.learning_rate = learning_rate
    
    def build(self):
        weights = {
        'c1_1': weight_variable([3, 3, 3, 64], name='W_conv11',trainable=False),
        'c1_2': weight_variable([3, 3, 64, 64], name='W_conv12',trainable=False),
        'c1_3': weight_variable([3, 3, 64, 64], name='W_conv13',trainable=False),
        
        'c2_1': weight_variable([3, 3, 64, 128], name='W_conv21',trainable=False),
        'c2_2': weight_variable([3, 3, 128, 128], name='W_conv22',trainable=False),
        'c2_3': weight_variable([3, 3, 128, 128], name='W_conv23',trainable=False),
        'c2_4': weight_variable([3, 3, 128, 128], name='W_conv24',trainable=False),
        'c2_5': weight_variable([3, 3, 128, 128], name='W_conv25',trainable=False),
        
        'c3_1': weight_variable([3, 3, 128, 256], name='W_conv31',trainable=False),
        'c3_2': weight_variable([3, 3, 256, 256], name='W_conv32',trainable=False),
        'c3_3': weight_variable([3, 3, 256, 256], name='W_conv33',trainable=False),
        'c3_4': weight_variable([3, 3, 256, 256], name='W_conv34',trainable=False),
        'c3_5': weight_variable([3, 3, 256, 256], name='W_conv35',trainable=False),
        'c3_6': weight_variable([3, 3, 256, 256], name='W_conv36',trainable=False),
        'c3_7': weight_variable([3, 3, 256, 256], name='W_conv37',trainable=False),
        'c3_8': weight_variable([3, 3, 256, 256], name='W_conv38',trainable=False),
        'c3_9': weight_variable([3, 3, 256, 256], name='W_conv39',trainable=False),
    
        'c4_1': weight_variable([3, 3, 256, 512], name='W_conv41',trainable=True),
        'c4_2': weight_variable([3, 3, 512, 512], name='W_conv42',trainable=True),
        'c4_3': weight_variable([3, 3, 512, 512], name='W_conv43',trainable=True),
        
        'fc5': weight_variable([512*7*6, 512], name='W_fc5',trainable=False),
        }

        global_step = tf.Variable(0, trainable=False)
        # input_image = tf.image.resize_images(images=self.input_x, size=(112, 96))
        conv11 = tf.nn.conv2d(self.input_x, weights['c1_1'], strides=[1, 2, 2, 1], padding='SAME', name='c_11')
        h_1 = prelu(conv11, name='act_1',trainable=False)

        # ResNet-1
        res_conv_11 = tf.nn.conv2d(h_1, weights['c1_2'], strides=[1,1,1,1], padding='SAME', name='res_c_12')
        res_h_11 = prelu(res_conv_11, name='res_act_11',trainable=False)
        res_conv_12 = tf.nn.conv2d(res_h_11, weights['c1_3'], strides=[1,1,1,1], padding='SAME', name='res_c_13')
        res_h_12 = prelu(res_conv_12, name='res_act_12',trainable=False)
        res_1 = h_1 + res_h_12

        # ResNet-2
        conv21 = tf.nn.conv2d(res_1, weights['c2_1'], strides=[1,2,2,1], padding='SAME', name='c_21')
        h_2 = prelu(conv21, name='act_2',trainable=False)
        res_conv_21 = tf.nn.conv2d(h_2, weights['c2_2'], strides=[1,1,1,1], padding='SAME', name='res_c_22')
        res_h_21 = prelu(res_conv_21, name='res_act_21',trainable=False)
        res_conv_22 = tf.nn.conv2d(res_h_21, weights['c2_3'], strides=[1,1,1,1], padding='SAME', name='res_c_23')
        res_h_22 = prelu(res_conv_22, name='res_act_22',trainable=False)
        res_21 = h_2 + res_h_22

        res_conv_23 = tf.nn.conv2d(res_21, weights['c2_4'], strides=[1,1,1,1], padding='SAME', name='res_c_24')
        res_h_23 = prelu(res_conv_23, name='res_act_23',trainable=False)
        res_conv_24 = tf.nn.conv2d(res_h_23, weights['c2_5'], strides=[1,1,1,1], padding='SAME', name='res_c_25')
        res_h_24 = prelu(res_conv_24, name='res_act_24',trainable=False)
        res_22 = res_21 + res_h_24

        # ResNet-3
        conv31 = tf.nn.conv2d(res_22, weights['c3_1'], strides=[1,2,2,1], padding='SAME', name='c_31')
        h_3 = prelu(conv31, name='act_3',trainable=False)
        res_conv_31 = tf.nn.conv2d(h_3, weights['c3_2'], strides=[1,1,1,1], padding='SAME', name='res_c_31')
        res_h_31 = prelu(res_conv_31, name='res_act_31',trainable=False)
        res_conv_32 = tf.nn.conv2d(res_h_31, weights['c3_3'], strides=[1,1,1,1], padding='SAME', name='res_c_32')
        res_h_32 = prelu(res_conv_32, name='res_act_32',trainable=False)
        res_31 = h_3 + res_h_32    

        res_conv_33 = tf.nn.conv2d(res_31, weights['c3_4'], strides=[1,1,1,1], padding='SAME', name='res_c_33')
        res_h_33 = prelu(res_conv_33, name='res_act_33',trainable=False)
        res_conv_34 = tf.nn.conv2d(res_h_33, weights['c3_5'], strides=[1,1,1,1], padding='SAME', name='res_c_34')
        res_h_34 = prelu(res_conv_34, name='res_act_34',trainable=False)
        res_32 = res_31 + res_h_34

        res_conv_35 = tf.nn.conv2d(res_32, weights['c3_6'], strides=[1,1,1,1], padding='SAME', name='res_c_35')
        res_h_35 = prelu(res_conv_31, name='res_act_35',trainable=False)
        res_conv_36 = tf.nn.conv2d(res_h_35, weights['c3_7'], strides=[1,1,1,1], padding='SAME', name='res_c_36')
        res_h_36 = prelu(res_conv_36, name='res_act_36',trainable=False)
        res_33 = res_32 + res_h_36

        res_conv_37 = tf.nn.conv2d(res_33, weights['c3_8'], strides=[1,1,1,1], padding='SAME', name='res_c_37')
        res_h_37 = prelu(res_conv_37, name='res_act_37',trainable=False)
        res_conv_38 = tf.nn.conv2d(res_h_37, weights['c3_9'], strides=[1,1,1,1], padding='SAME', name='res_c_38')
        res_h_38 = prelu(res_conv_38, name='res_act_38',trainable=False)
        res_34 = res_33 + res_h_38

        # ResNet-4
        conv41 = tf.nn.conv2d(res_34, weights['c4_1'], strides=[1,2,2,1], padding='SAME', name='c_41' )
        h_4 = prelu(conv41, name='act_4',trainable=True)
        res_conv_41 = tf.nn.conv2d(h_4, weights['c4_2'], strides=[1,1,1,1], padding='SAME', name='res_c_41')
        res_h_41 = prelu(res_conv_41, name='res_act_41',trainable=True)
        res_conv_42 = tf.nn.conv2d(res_h_41, weights['c4_3'], strides=[1,1,1,1], padding='SAME', name='res_c_42')
        res_h_42 = prelu(res_conv_42, name='res_act_42',trainable=True)
        res_41 = h_4 + res_h_42

        flat1 = tf.layers.flatten(res_41, 'flat_1')
        fc_1 = tf.layers.dense(flat1, 512, name='fc_1')

        logits = tf.layers.dense(fc_1, self.nb_classes, name='pred_logits')
        output_pred = tf.nn.softmax(logits, name='output',axis=1)
        loss_val = ce_loss(logits, self.input_label)

        train_op = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).minimize(loss_val)
        # correct = tf.equal(tf.argmax(output_pred, 1), tf.argmax(input_y, 1))
        # correct = tf.equal(tf.cast(tf.argmax(output_pred, 1), tf.int32), self.input_label)
        # acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        return fc_1, loss_val, train_op

#generate pb
tf.reset_default_graph()
model = Sphere(nb_classes=10575, scale=True, height=112, width=96)
embedding, loss, optimizer = model.build()
print(embedding)
training_epochs = 5
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
graph_def = sess.graph.as_graph_def()
json_string = json_format.MessageToJson(graph_def)
obj = json.loads(json_string)
tf.compat.v1.train.write_graph(sess.graph, "./", "sphere_frozen123.pb", as_text=False)


#generate txt
trainable_var = tf.trainable_variables()
global_var = tf.global_variables()
with open("sphere2_trainable_var_f123.txt", "a+") as f:
    variables_sum = 0
    for var in trainable_var:
        accumulate = 1
        for i in range(len(var.shape)):
            accumulate = var.shape[i] * accumulate
        variables_sum = accumulate + variables_sum
        f.write(var.op.name + ":" + str(var.shape) + "\n")
    print(variables_sum)

with open("sphere2_trainable_init_var_f123.txt", "a+") as f:
    variables_sum = 0
    for var in global_var:
        accumulate = 1
        for i in range(len(var.shape)):
            accumulate = var.shape[i] * accumulate
        variables_sum = accumulate + variables_sum
        f.write(var.initial_value.op.name + ":" + str(var.shape) + "\n")
    print(variables_sum)

with open("sphere2_feed_fetch_f123.txt", "a+") as f:
    f.write(model.input_label.op.name + ":" + str(model.input_label.shape) + "\n")
    f.write(model.input_x.op.name + ":" + str(model.input_x.shape) + "\n")
    f.write(init.name + ":" + "---" + "\n")
    f.write(optimizer.name + ":" + "---" + "\n")
    f.write(loss.name + ":" + "---" + "\n")