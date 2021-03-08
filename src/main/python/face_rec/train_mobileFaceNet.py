# -*- coding: utf-8 -*-
# /usr/bin/env/python3

"""
TensorFlow implementation for MobileFaceNet.
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics

from losses.face_losses import insightface_loss, cosineface_loss, combine_loss
from nets.MobileFaceNet import inference
from utils.common import train
from utils.data_process import load_data
from verification import evaluate
from utils.write_pb_mobile import write_graph_netInfo

slim = tf.contrib.slim

prj_path="/data/zhangxun/federated_app/src/main/python/face_rec"


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--summary_path', default=prj_path+'/output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default=prj_path+'/output/ckpt', help='the ckpt file save path')
    parser.add_argument('--ckpt_best_path', default=prj_path+'/output/ckpt_best', help='the best ckpt file save path')
    parser.add_argument('--log_file_path', default=prj_path+'/output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_max_keep', default=50, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--summary_interval', default=400, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=50, help='intervals to save ckpt file')
    parser.add_argument('--pre_trained_model', type=str, default='',
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.999)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the pre_logits layer.', default=2e-5)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for pre_logits norm loss.', default=1.0)
    parser.add_argument('--loss_type', default='insightface',
                        help='loss type, choice type are insightface/cosine/combine')
    parser.add_argument('--margin_s', type=float,
                        help='insightface_loss/cosineface_losses/combine_loss loss scale.', default=64.)
    parser.add_argument('--margin_m', type=float,
                        help='insightface_loss/cosineface_losses/combine_loss loss margin.', default=0.5)
    parser.add_argument('--margin_a', type=float,
                        help='combine_loss loss margin a.', default=1.0)
    parser.add_argument('--margin_b', type=float,
                        help='combine_loss loss margin b.', default=0.2)
    ######################################################################################################
    parser.add_argument('--max_epoch', default=30, help='epoch to train the network')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--img_txt',
                        type=str,
                        default='/data/zhangxun/data/CASIA-WebFace-aligned/imgs.txt',
                        help='combine_loss loss margin b.')
    parser.add_argument('--img_root_path',
                        type=str,
                        default='/data/zhangxun/data/CASIA-WebFace-aligned/imgs',
                        help='combine_loss loss margin b.')
    parser.add_argument('--class_number', type=int, default=1006,
                        help='class number depend on your training data sets, '
                             'MS1M-V1: 85164 ,'
                             'MS1M-V2: 85742 ,'
                             'CASIA-WebFace:1006')
    parser.add_argument('--dataset_name', type=str,
                        default="CASIA_WebFace",
                        help='dataset name')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.',
                        default=128)
    parser.add_argument('--weight_decay',
                        default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--lr_schedule', help='Number of epochs for learning rate piecewise.',
                        default=[4, 7, 9, 11])
    parser.add_argument('--train_batch_size',
                        default=90, help='batch size to train network')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.',
                        default=100)
    parser.add_argument('--eval_data_sets',
                        default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'],
                        help='evaluation datasets')
    parser.add_argument('--eval_db_path',
                        default='/data/zhangxun/data/evaluation',
                        help='evaluate datasets base path')
    parser.add_argument('--eval_nrof_folds',
                        type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.',
                        default=10)
    parser.add_argument('--tf_records_file_path',
                        default='/data/zhangxun/data/faces_ms1m_112x112/tfrecords',
                        type=str,
                        help='path to the output of tf records file path')
    parser.add_argument('--graph_path', type=str,
                        default=prj_path+'/graph/mobileFaceNet',
                        help='graph path.')
    parser.add_argument('--model_name', type=str,
                        default='mobileFaceNet',
                        help='model name:'
                             'mobileFaceNet')
    parser.add_argument('--only_gen_graph', type=bool,
                        default=True,
                        help='generate graph ')

    args = parser.parse_args()
    return args


def read_img(batch_img_pt):
    size = len(batch_img_pt)
    img = np.zeros(shape=(size, 112, 112, 3), dtype=np.float32)
    label = np.zeros(shape=(size,), dtype=np.int64)
    for idx, path in enumerate(batch_img_pt):
        tmp = os.path.join(args.img_root_path,str(path.replace("\n", "")))
        img[idx] = np.array(Image.open(tmp).resize((112, 112)))
        label[idx] = path.split("/")[0]
    img = (img - 127.5) / 128
    return img, label


if __name__ == '__main__':
    mobile = False
    with tf.Graph().as_default():
        args = get_parser()

        # create log dir
        subdir = args.dataset_name + "_" + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.log_file_path), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)
        summary_dir = os.path.join(args.summary_path, args.dataset_name)

        # define global parameters
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        epoch = tf.Variable(name='epoch', initial_value=-1, trainable=False)
        # define placeholder
        inputs = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
        labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None,
                                                              name='phase_train')

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        pre_logits, net_points = inference(inputs,
                                           bottleneck_layer_size=args.embedding_size,
                                           phase_train=phase_train_placeholder,
                                           weight_decay=args.weight_decay,
                                           mobile=mobile)

        embeddings = tf.nn.l2_normalize(pre_logits, 1, 1e-10, name='embeddings')

        # Norm for the pre_logits
        eps = 1e-5
        pre_logits_norm = tf.reduce_mean(tf.norm(tf.abs(pre_logits) + eps, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, pre_logits_norm * args.prelogits_norm_loss_factor)

        # inference_loss, logit = cos_loss(pre_logits, labels, args.class_number)
        w_init_method = slim.initializers.xavier_initializer()
        if args.loss_type == 'insightface':
            inference_loss, logit = insightface_loss(embeddings, labels, args.class_number, w_init_method)
        elif args.loss_type == 'cosine':
            inference_loss, logit = cosineface_loss(embeddings, labels, args.class_number, w_init_method)
        elif args.loss_type == 'combine':
            inference_loss, logit = combine_loss(embeddings, labels, args.train_batch_size, args.class_number,
                                                 w_init_method)
        else:
            assert 0, 'loss type error, choice item just one of [insightface, cosine, combine], please check!'
        tf.add_to_collection('losses', inference_loss)

        # total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([inference_loss] + regularization_losses, name='total_loss')

        # define the learning rate schedule
        learning_rate = tf.train.piecewise_constant(epoch, boundaries=args.lr_schedule,
                                                    values=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                                                    name='lr_schedule')

        # define sess
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping,
                                gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # calculate accuracy
        pred = tf.nn.softmax(logit)
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
        Accuracy_Op = tf.reduce_mean(correct_prediction)

        # summary writer
        summary = tf.summary.FileWriter(summary_dir, sess.graph)
        summaries = [tf.summary.scalar('inference_loss', inference_loss),
                     tf.summary.scalar('total_loss', total_loss),
                     tf.summary.scalar('leraning_rate', learning_rate)]

        # add train info to tensor board summary
        summary_op = tf.summary.merge(summaries)

        # train op
        train_op = train(total_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay,
                         tf.global_variables(), summaries, args.log_histograms)
        inc_global_step_op = tf.assign_add(global_step, 1, name='increment_global_step')
        inc_epoch_op = tf.assign_add(epoch, 1, name='increment_epoch')

        # saver to load pre trained model or save model
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_max_keep)

        # init all variables
        global_var_init = tf.global_variables_initializer()
        local_var_init = tf.local_variables_initializer()
        sess.run(global_var_init)
        sess.run(local_var_init)

        net_info = {"x": inputs,
                    "y": labels,
                    "global_var_init": global_var_init,
                    "train_op": train_op,
                    "loss_op": total_loss,
                    "accuracy_op": Accuracy_Op}

        write_graph_netInfo(args, sess, net_info, "MobileFaceNet")

        if args.only_gen_graph:
            exit()

        # record trainable variable
        hd = open("model/txt/trainable_var.txt", "w")
        for var in tf.trainable_variables():
            hd.write(str(var))
            hd.write('\n')
        hd.close()

        # record the network architecture
        hd = open("model/txt/MobileFaceNet_Arch.txt", 'w')
        for key in net_points.keys():
            info = '{}:{}\n'.format(key, net_points[key].get_shape().as_list())
            hd.write(info)
        hd.close()

        # prepare train dataset
        # the image is substracted 127.5 and multiplied 1/128.
        with open(args.img_txt, "r") as f:
            img_txt_pt = f.readlines()
            chunks_img = [img_txt_pt[i:i + args.train_batch_size]
                          for i in range(0, len(img_txt_pt), args.train_batch_size)]

        # pre_trained model path
        pre_trained_model = None
        if args.pre_trained_model:
            pre_trained_model = os.path.expanduser(args.pre_trained_model)
            print('Pre-trained model: %s' % pre_trained_model)

        # load pre trained model
        if pre_trained_model:
            print('Restoring pre trained model: %s' % pre_trained_model)
            ckpt = tf.train.get_checkpoint_state(pre_trained_model)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # output file path
        if not os.path.exists(args.log_file_path):
            os.makedirs(args.log_file_path)
        if not os.path.exists(args.ckpt_best_path):
            os.makedirs(args.ckpt_best_path)

        # prepare validate datasets
        ver_list = []
        ver_name_list = []
        for db in args.eval_data_sets:
            print('begin db %s convert.' % db)
            data_set = load_data(db, args.image_size, args)
            ver_list.append(data_set)
            ver_name_list.append(db)

        count = 0
        total_accuracy = {}
        for i in range(args.max_epoch):
            _ = sess.run(inc_epoch_op)
            for j in chunks_img:
                images_train, labels_train = read_img(j)
                feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                start = time.time()
                _, total_loss_val, inference_loss_val, reg_loss_val, _, acc_val = \
                    sess.run([train_op, total_loss, inference_loss, regularization_losses, inc_global_step_op,
                              Accuracy_Op],
                             feed_dict=feed_dict)
                end = time.time()
                pre_sec = args.train_batch_size / (end - start)

                count += 1
                # print training information
                if count > 0 and count % args.show_info_interval == 0:
                    print(
                        'epoch %d, total_step %d, total loss is %.2f , '
                        'inference loss is %.2f, reg_loss is %.2f, '
                        'training accuracy is %.6f, time %.3f samples/sec' %
                        (i, count, total_loss_val, inference_loss_val, np.sum(reg_loss_val), acc_val, pre_sec))

                # save summary
                if count > 0 and count % args.summary_interval == 0:
                    feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                    summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                    summary.add_summary(summary_op_val, count)

                # save ckpt files
                if count > 0 and count % args.ckpt_interval == 0:
                    filename = 'MobileFaceNet_iter_{:d}'.format(count) \
                               + "_" + args.dataset_name + '.ckpt'
                    filename = os.path.join(args.ckpt_path, filename)
                    saver.save(sess, filename)

                # validate
                if count > 0 and count % args.validate_interval == 0:
                    print('\nIteration', count, 'testing...')
                    for db_index in range(len(ver_list)):
                        start_time = time.time()
                        data_sets, is_same_list = ver_list[db_index]
                        emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
                        nrof_batches = data_sets.shape[0] // args.test_batch_size
                        for index in range(nrof_batches):  # actual is same multiply 2, test data total
                            start_index = index * args.test_batch_size
                            end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                            feed_dict = {inputs: data_sets[start_index:end_index, ...],
                                         phase_train_placeholder: False}
                            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                        tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, is_same_list,
                                                                         nrof_folds=args.eval_nrof_folds)
                        duration = time.time() - start_time

                        print("total time %.3fs to evaluate %d images of %s" % (
                            duration, data_sets.shape[0], ver_name_list[db_index]))
                        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                        print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))

                        auc = metrics.auc(fpr, tpr)
                        print('Area Under Curve (AUC): %1.3f' % auc)
                        eer = brentq(lambda x: 1. - x - interpolate.interp1d(
                            fpr, tpr, fill_value="extrapolate")(x), 0., 1.)
                        print('Equal Error Rate (EER): %1.3f\n' % eer)

                        with open(os.path.join(log_dir, '{}_result.txt'.format(ver_name_list[db_index])),
                                  'at') as f:
                            f.write('%d\t%.5f\t%.5f\n' % (count, np.mean(accuracy), val))

                        if ver_name_list == 'lfw' and np.mean(accuracy) > 0.992:
                            print('best accuracy is %.5f' % np.mean(accuracy))
                            filename = 'MobileFaceNet_iter_best_{:d}'.format(count) + \
                                       "_" + args.dataset_name + '.ckpt'
                            filename = os.path.join(args.ckpt_best_path, filename)
                            saver.save(sess, filename)
