import argparse

import numpy as np
import tensorflow as tf
from sklearn import metrics

from nets import SphereFaceNet
from utils.data_process import load_val_data, get_img_path_and_label, get_data, random_mini_batches
from utils.write_pb_mobile import write_graph_netInfo
from verification import evaluate

prj_path="/Users/voyager/code/android-demo/federated_app/src/main/python/face_rec"


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--image_size', default=[112, 96], help='the image size')
    parser.add_argument('--class_number', type=int, default=1006,
                        help='class number depend on your training datasets, '
                             'MS1M-V1: 85164,'
                             'MS1M-V2: 85742'
                             'CASIA-WebFace-aligned: 1006')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=1)
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default='lfw', help='evluation datasets')
    parser.add_argument('--eval_db_path', default='/data/zhangxun/data/evaluation', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoints_dir', default='output/SphereFace', help='model path')
    parser.add_argument('--data_path', default='/data/zhangxun/data/CASIA-WebFace-aligned/imgs', help='model path')
    parser.add_argument('--only_gen_graph', type=bool,
                        default=True,
                        help='generate graph ')
    parser.add_argument('--graph_path', type=str,
                        default=prj_path + '/graph/sphereFaceNet',
                        help='graph path.')
    parser.add_argument('--model_name', type=str,
                        default='sphereFaceNet',
                        help='model name:'
                             'sphereFaceNet')
    args = parser.parse_args()
    return args


def ce_loss(logits, label):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    return cross_entropy_loss


def train_pipe():
    args = get_parser()
    input_x = tf.placeholder(tf.float32, [None, args.image_size[0], args.image_size[1], 3], name='input_x')
    input_y = tf.placeholder(tf.int32, [None, args.class_number], name='input_y')
    # learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')

    model = SphereFaceNet.Model(input_x)

    prelogits = tf.layers.dense(model.embeddings, args.class_number, name='prelogits')
    pre_softmax = tf.nn.softmax(prelogits, name='output', axis=1)
    loss = ce_loss(prelogits, input_y)

    # train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    train_op = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9).minimize(loss)
    correct = tf.equal(tf.argmax(pre_softmax, 1), tf.argmax(input_y, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    # correct = tf.equal(tf.cast(tf.argmax(pre_softmax, 1), tf.int32), tf.cast(tf.argmax(input_y, 1), tf.int32))
    # acc = tf.reduce_mean(tf.cast(correct, tf.float32))

    embeddings_l2 = tf.nn.l2_normalize(model.embeddings, 1, 1e-10, name='embeddings')

    return input_x, input_y, train_op, loss, acc, embeddings_l2


if __name__ == '__main__':
    args = get_parser()
    save_graph = True
    with tf.Graph().as_default():
        input_x, input_y, train_op, loss, acc, embeddings_l2 = train_pipe()

        global_var_init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(global_var_init)
        net_info = {"x": input_x,
                    "y": input_y,
                    "global_var_init": global_var_init,
                    "train_op": train_op,
                    "loss_op": loss,
                    "accuracy_op": acc}

        write_graph_netInfo(args, sess, net_info, None)

        if args.only_gen_graph:
            trainable_var = tf.trainable_variables()
            for var in trainable_var:
                try:
                    v = sess.run(var)
                    v_ravel = v.ravel()
                    np.save("./weights_numpy/" +
                            var.op.name.replace("/", "_") + ".npy", v)
                    with open("./weights_numpy/" +
                              var.op.name.replace("/", "_") + ".txt", "w") as f:
                        for i in v_ravel:
                            f.write(str(float(i)))
                            f.write("\n")

                    # print(var.initial_value.op.name + ";" + str(var.shape) + "\n")
                except Exception as e:
                    print(e)
            exit()



        data_block = get_img_path_and_label(args.data_path, 10)
        saver = tf.train.Saver(max_to_keep=5)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stop = False
            data_sets, issame_list = load_val_data(args.eval_datasets, args.image_size, args)
            for epoch in range(0, 100):

                #############################################################
                # eval data
                val_em_all = np.zeros((12000, 512))
                for i in range(120):
                    ds = data_sets[i * 100: (i + 1) * 100]
                    val_feed_dict = {input_x: ds}
                    val_em = sess.run([embeddings_l2], feed_dict=val_feed_dict)
                    val_em_all[i * 100: (i + 1) * 100] = val_em[0]
                tpr, fpr, accuracy, _, _, _ = evaluate(val_em_all, issame_list, nrof_folds=10)
                auc = metrics.auc(fpr, tpr)
                print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))
                print('Area Under Curve (AUC): %1.3f' % auc)
                #############################################################

                # training
                step = 0
                for block_idx in range(len(data_block)):
                    block = data_block[block_idx]
                    x, y = get_data(block, args.class_number)
                    mini_batch = random_mini_batches(x, y, args.batch_size)
                    for batch in mini_batch:
                        x_batch, y_batch = batch
                        train_feed_dict = {input_x: (x_batch - 127.5) / 128, input_y: y_batch}
                        _, train_loss, train_acc = sess.run([train_op, loss, acc], train_feed_dict)
                        step += 1
                        if step % 50 == 0:
                            print('%d epoch %d step train loss : %f acc: %f' % (epoch, step, train_loss, train_acc))
                        if step % 2000 == 0:
                            path = saver.save(sess, args.checkpoints_dir + '/SphereFace.ckpt', global_step=step)
                            print('Save model in step:{}, path:{}'.format(step, path))
