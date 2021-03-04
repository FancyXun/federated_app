import numpy as np
import tensorflow as tf
from sklearn import metrics

from train_SphereFaceNet import train_pipe, get_parser
from utils.data_process import load_val_data, get_img_path_and_label, get_data, random_mini_batches
from verification import evaluate

if __name__ == '__main__':
    args = get_parser()
    client = 3
    update_weights = [[] for _ in range(client)]
    clients_weights = [[] for _ in range(client)]
    data_block = get_img_path_and_label(args.data_path, 10)
    data_block_per_client = len(data_block) // client
    data_sets, issame_list = load_val_data(args.eval_datasets, args.image_size, args)
    for epoch in range(0, 100):
        tf.contrib.keras.backend.clear_session()
        eval_flag = True
        with tf.Graph().as_default():
            input_x, input_y, train_op, loss, acc, embeddings_l2 = train_pipe()
            saver = tf.train.Saver(max_to_keep=5)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for client_id in range(client):
                    step = 0
                    if epoch == 0:
                        sess.run(tf.global_variables_initializer())
                    else:
                        trainable_var = tf.trainable_variables()
                        for var, c_w in zip(trainable_var, zip(*update_weights)):
                            mean_val = 0
                            for j in c_w:
                                mean_val = mean_val + j
                            mean_val = mean_val / len(c_w)
                            mean_val_op = var.assign(mean_val)
                            sess.run(mean_val_op)
                    if eval_flag:
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
                        eval_flag = False

                    client_data_block = data_block[
                                        client_id * data_block_per_client: (client_id + 1) * data_block_per_client]
                    for block_idx in range(len(client_data_block)):
                        block = client_data_block[block_idx]
                        x, y = get_data(block, args.class_number)
                        mini_batch = random_mini_batches(x, y, args.batch_size)
                        for batch in mini_batch:
                            x_batch, y_batch = batch
                            train_feed_dict = {input_x: (x_batch - 127.5) / 128, input_y: y_batch}
                            _, train_loss, train_acc = sess.run([train_op, loss, acc], train_feed_dict)
                            step += 1
                            if step % 50 == 0:
                                print('%d epoch %d step train loss : %f acc: %f' % (epoch, step, train_loss, train_acc))
                                with open("client" + str(client_id) + ".txt", "a+") as f:
                                    f.write('%d epoch %d step train loss : %f acc: %f' %
                                            (epoch, step, train_loss, train_acc))
                                    f.write("\n")
                            if step % 2000 == 0:
                                path = saver.save(sess, args.checkpoints_dir + '/model.ckpt', global_step=step)
                                print('Save model in step:{}, path:{}'.format(step, path))

                    trainable_var = tf.trainable_variables()
                    c = []
                    noise_scaling_parameter = 1e3
                    for var in trainable_var:
                        tmp = sess.run(var)
                        tmp += tf.random.normal(
                            tmp.shape, stddev=tf.reduce_mean(tmp) / noise_scaling_parameter)
                        c.append(tmp)
                    clients_weights[client_id] = c
                for idx, weights_cli in enumerate(clients_weights):
                    update_weights[idx] = weights_cli
