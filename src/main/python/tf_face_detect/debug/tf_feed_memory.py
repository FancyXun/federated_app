import argparse
from os import getpid

import numpy as np
import psutil
import tensorflow as tf


def fc(inputs, output_size):
    with tf.variable_scope("FC"):
        input_size = inputs.get_shape()[-1].value
        W = tf.get_variable("W", shape=[input_size, output_size])
        b = tf.get_variable("b", shape=[output_size], initializer=tf.constant_initializer(0))
        out = tf.nn.xw_plus_b(inputs, W, b)
    return out


def create_model(input_size, output_size):
    # model placeholders:
    with tf.variable_scope("Inputs"):
        input_placeholder = tf.placeholder(
            tf.float32, [None, input_size], name="input_placeholder"
        )
    # meaningless function of inputs
    op = tf.reduce_mean(tf.reduce_sum(fc(input_placeholder, output_size), 1))
    return input_placeholder, op


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=7000)
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--output_size', type=int, default=100)
    parser.add_argument('--device', type=str, default="gpu:0")
    return parser.parse_args(args=args)


def create_batches(inputs, input_size, batch_size, n):
    batches = []
    for i in range(n):
        X = np.random.uniform(-1.0, 1.0, size=(batch_size, input_size))
        batches.append({inputs: X})
    return batches


def main():
    args = parse_args()
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    np.random.seed(1234)
    process = psutil.Process(getpid())

    with tf.Session(config=session_conf) as session, tf.device(args.device):
        inputs, op = create_model(args.input_size, args.output_size)
        session.run(tf.global_variables_initializer())
        batches = create_batches(inputs, args.input_size, args.batch_size, 20)

        for epoch in range(args.max_epochs):
            before = process.memory_percent()
            for feed_dict in batches:
                session.run(op, feed_dict)
            after = process.memory_percent()
            print("MEMORY CHANGE %.4f -> %.4f" % (before, after))


if __name__ == "__main__":
    main()