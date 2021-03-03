import os
import tensorflow as tf
from nets.MobileFaceNet import inference

training_checkpoint = "../arch/pretrained_model/MobileFaceNet_TF.ckpt"
output_dir = '../model/pb/mobileFaceNet_pts.pb'


def freeze_graph_def(sess, output_node_names):
    # Replace all the variables in the graph with constants of the same values
    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
    sess, sess.graph_def, output_node_names.split(","))
    return output_graph_def


def save():
    data_input = tf.placeholder(name='input', dtype=tf.float32, shape=[None, 112, 112, 3])
    output, _ = inference(data_input, bottleneck_layer_size=192) # ? why 192
    tf.identity(output, name='embeddings')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, training_checkpoint)

        # Freeze the graph def
        output_graph_def = freeze_graph_def(sess, 'embeddings')

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_dir, 'wb') as f:
            f.write(output_graph_def.SerializeToString())


save()