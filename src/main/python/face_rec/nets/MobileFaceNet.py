# Copyright 2018 The AI boy xsr-ai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""MobileFaceNets.

MobileFaceNets, which use less than 1 million parameters and are specifically tailored for high-accuracy real-time
face verification on mobile and embedded devices.

here is MobileFaceNets architecture, reference from MobileNet_V2 (https://github.com/xsr-ai/MobileNetv2_TF).

As described in https://arxiv.org/abs/1804.07573.

  MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices

  Sheng Chen, Yang Liu, Xiang Gao, Zhen Han

"""

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import namedtuple

import tensorflow as tf

slim = tf.contrib.slim

# Conv and InvResBlock namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# InvResBlock defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'ratio'])
DepthwiseConv = namedtuple('DepthwiseConv', ['kernel', 'stride', 'depth', 'ratio'])
InvResBlock = namedtuple('InvResBlock', ['kernel', 'stride', 'depth', 'ratio', 'repeate'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=64, ratio=1),
    DepthwiseConv(kernel=[3, 3], stride=1, depth=64, ratio=1),

    InvResBlock(kernel=[3, 3], stride=2, depth=64, ratio=2, repeate=5),
    InvResBlock(kernel=[3, 3], stride=2, depth=128, ratio=4, repeate=1),
    InvResBlock(kernel=[3, 3], stride=1, depth=128, ratio=2, repeate=6),
    InvResBlock(kernel=[3, 3], stride=2, depth=128, ratio=4, repeate=1),
    InvResBlock(kernel=[3, 3], stride=1, depth=128, ratio=2, repeate=2),

    Conv(kernel=[1, 1], stride=1, depth=512, ratio=1),
]

initial = tf.glorot_uniform_initializer()


def inverted_block(net, input_filters, output_filters, expand_ratio, stride, scope=None, mobile=False, idx=0):
    '''fundamental network struture of inverted residual block'''
    with tf.name_scope(scope):
        res_block = slim.conv2d(inputs=net, num_outputs=input_filters * expand_ratio, kernel_size=[1, 1])
        # depthwise conv2d
        if mobile:
            input_channel = int(res_block.get_shape()[-1])
            res_block = tf.nn.depthwise_conv2d(res_block,
                                         tf.get_variable("res_block_depthwise_"+str(idx),
                                                         shape=[3, 3] +
                                                               [input_channel, 1],
                                                         initializer=initial),
                                         strides=[1, stride, stride, 1],
                                         padding="SAME")
            # res_block = separable_conv2d_mobile(res_block, [3, 3], stride, name="res_block_depthwise_"+str(idx))
            # res_block = slim.conv2d(net, int(net.get_shape()[-1]), [3, 3],
            #                         stride=stride, normalizer_fn=slim.batch_norm)
        else:
            res_block = slim.separable_conv2d(inputs=res_block, num_outputs=None, kernel_size=[3, 3], stride=stride,
                                              depth_multiplier=1.0, normalizer_fn=slim.batch_norm)
        res_block = slim.conv2d(inputs=res_block, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
        # stride 2 blocks
        if stride == 2:
            return res_block
        # stride 1 block
        else:
            if input_filters != output_filters:
                net = slim.conv2d(inputs=net, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
            return tf.add(res_block, net)


def mobilenet_v2_base(inputs,
                      final_endpoint='Conv2d_7',
                      min_depth=8,
                      conv_defs=None,
                      scope=None,
                      mobile=False):
    """Mobilenet v2.

    Constructs a Mobilenet v2 network from inputs to the given final endpoint.

    Args:
      inputs: a tensor of shape [batch_size, height, width, channels].
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of ['Conv2d_0', 'Conv2d_1_InvResBlock', 'Conv2d_2_InvResBlock',
        'Conv2d_3_InvResBlock', 'Conv2d_4_InvResBlock', 'Conv2d_5_InvResBlock,
        'Conv2d_6_InvResBlock', 'Conv2d_7_InvResBlock', 'Conv2d_8'].
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced output depth to min_depth.
      conv_defs: A list of ConvDef namedtuples specifying the net architecture.
      scope: Optional variable_scope.
      mobile: Using mobile or not

    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.

    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values
                  is not allowed.
    """
    depth = lambda d: max(int(d), min_depth)
    end_points = {}
    block_idx = 0

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    with tf.variable_scope(scope, 'MobileFaceNet', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):

            net = inputs
            for i, conv_def in enumerate(conv_defs):
                end_point_base = 'Conv2d_%d' % i

                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                                      stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                elif isinstance(conv_def, DepthwiseConv):
                    end_point = 'DepthwiseConv'
                    # depthwise conv2d
                    # 56 56 64 -> 56 56 64
                    '''
                    Adds a depth-separable 2D convolution with optional batch_norm layer.

                    This op first performs a depthwise convolution that acts separately on
                    channels, creating a variable called `depthwise_weights`. If `num_outputs`
                    is not None, it adds a pointwise convolution that mixes channels, creating a
                    variable called `pointwise_weights`. Then, if `normalizer_fn` is None,
                    it adds bias to the result, creating a variable called 'biases', otherwise,
                    the `normalizer_fn` is applied. It finally applies an activation function
                    to produce the end result.
                    '''
                    if mobile:
                        input_channel = int(net.get_shape()[-1])
                        net = tf.nn.depthwise_conv2d(net,
                                                     tf.get_variable("DepthwiseConv",
                                                                     shape=conv_def.kernel +
                                                                           [input_channel, 1],
                                                                     initializer=initial),
                                                     strides=[1, conv_def.stride,
                                                              conv_def.stride, 1],
                                                     padding="SAME")
                        # net = separable_conv2d_mobile(net, conv_def.kernel, conv_def.stride, name="DepthwiseConv")
                        # net = slim.conv2d(net, int(net.get_shape()[-1]), conv_def.kernel,
                        # stride=conv_def.stride, normalizer_fn=slim.batch_norm)
                    else:
                        net = slim.separable_conv2d(inputs=net, num_outputs=None, kernel_size=conv_def.kernel,
                                                    stride=conv_def.stride,
                                                    depth_multiplier=1.0, normalizer_fn=slim.batch_norm)
                    net = slim.conv2d(inputs=net, num_outputs=conv_def.depth, kernel_size=[1, 1], activation_fn=None)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                elif isinstance(conv_def, InvResBlock):
                    end_point = end_point_base + '_InvResBlock'
                    # inverted bottleneck blocks
                    input_filters = net.shape[3].value
                    # first layer needs to consider stride
                    net = inverted_block(net, input_filters, depth(conv_def.depth), conv_def.ratio, conv_def.stride,
                                         end_point + '_0', mobile=mobile, idx=block_idx)
                    block_idx += 1
                    for index in range(1, conv_def.repeate):
                        suffix = '_' + str(index)
                        net = inverted_block(net, input_filters, depth(conv_def.depth), conv_def.ratio, 1,
                                             end_point + suffix, mobile=mobile, idx=block_idx)
                        block_idx += 1
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def mobilenet_v2(inputs,
                 bottleneck_layer_size=128,
                 is_training=False,
                 min_depth=8,
                 conv_defs=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobileFaceNet',
                 global_pool=False,
                 mobile=False):
    """Mobilenet v2 model for classification.

    Args:
      inputs: a tensor of shape [batch_size, height, width, channels].
      bottleneck_layer_size: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
      is_training: whether is training or not.
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced output depth to min_depth..
      conv_defs: A list of ConvDef namedtuples specifying the net architecture.
      spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
          of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
      global_pool: Optional boolean flag to control the avgpooling before the
        logits layer. If false or unset, pooling is done with a fixed window
        that reduces default-sized inputs to 1x1, while larger inputs lead to
        larger outputs. If true, any input size is pooled down to 1x1.
      mobile: Using mobile or not

    Returns:
      net: a 2D Tensor with the logits (pre-softmax activations) if bottleneck_layer_size
        is a non-zero integer, or the non-dropped-out input to the logits layer
        if bottleneck_layer_size is 0 or None.
      end_points: a dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: Input rank is invalid.
    """
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))

    with tf.variable_scope(scope, 'MobileFaceNet', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = mobilenet_v2_base(inputs, scope=scope,
                                                min_depth=min_depth,
                                                conv_defs=conv_defs, mobile=mobile)

            with tf.variable_scope('Logits'):
                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                else:
                    # Pooling with a fixed kernel size.
                    kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])

                    # Global depthwise conv2d
                    if mobile:
                        # net = separable_conv2d_mobile(net, kernel_size, stride=1,
                        #                               padding='VALID', name="Globaldepthwise")
                        input_channel = int(net.get_shape()[-1])
                        net = tf.nn.depthwise_conv2d(net,
                                                     tf.get_variable("Globaldepthwise",
                                                                     shape=kernel_size +
                                                                           [input_channel, 1],
                                                                     initializer=initial),
                                                     strides=[1, 1, 1, 1],
                                                     padding="VALID")
                        # net = slim.conv2d(net, int(net.get_shape()[-1]), kernel_size, stride=1, padding='VALID')
                    else:
                        net = slim.separable_conv2d(inputs=net, num_outputs=None, kernel_size=kernel_size, stride=1,
                                                    depth_multiplier=1.0, activation_fn=None, padding='VALID')
                    net = slim.conv2d(inputs=net, num_outputs=512, kernel_size=[1, 1], stride=1, activation_fn=None,
                                      padding='VALID')
                    end_points['GDConv'] = net

                if not bottleneck_layer_size:
                    return net, end_points
                # 1 x 1 x 1024
                # net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                logits = slim.conv2d(net, bottleneck_layer_size, kernel_size=[1, 1], stride=1, activation_fn=None,
                                     scope='LinearConv1x1')

                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits

    return logits, end_points


mobilenet_v2.default_image_size = 112


# too slow
def separable_conv2d_mobile(net, kernel, stride, padding='SAME', name=None):
    net_list = []
    channel_dim = int(net.get_shape()[-1])
    filter_all = tf.get_variable(name, shape=kernel + [channel_dim, 1, 1], initializer=initial)
    net = tf.expand_dims(net, axis=4)
    print(channel_dim)
    for channel in range(channel_dim):
        net_fm = net[:, :, :, channel]
        # net_list.append(slim.conv2d(net_fm, 1, kernel, stride=stride,
        #                             normalizer_fn=slim.batch_norm, padding=padding))
        net_list.append(
                tf.nn.conv2d(
                    net_fm, filter_all[:, :, channel],
                    strides=[stride, stride, stride, stride],
                    padding=padding))
    net = tf.nn.bias_add(tf.squeeze(tf.stack(net_list, axis=3), axis=4),
                         tf.get_variable(name+"_bias", shape=[channel_dim], initializer=initial))
    return net


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are large enough.

    Args:
      input_tensor: input tensor of size [batch_size, height, width, channels].
      kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
      a tensor with the kernel size.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out


def prelu(input, name=''):
    alphas = tf.get_variable(name=name + 'prelu_alphas',
                             initializer=tf.constant(0.25, dtype=tf.float32, shape=[input.get_shape()[-1]]))
    pos = tf.nn.relu(input)
    neg = alphas * (input - abs(input)) * 0.5
    return pos + neg


def mobilenet_v2_arg_scope(is_training=True,
                           weight_decay=0.00005,
                           regularize_depthwise=False):
    """Defines the default MobilenetV2 arg scope.

    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      regularize_depthwise: Whether or not apply regularization on depthwise.

    Returns:
      An `arg_scope` to use for the mobilenet v2 model.
    """
    batch_norm_params = {
        'is_training': is_training,
        'trainable': False,
        'center': True,
        'scale': True,
        'fused': True,
        'decay': 0.995,
        'epsilon': 2e-5,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    # Set weight_decay for weights in Conv and InvResBlock layers.
    # weights_init = tf.truncated_normal_initializer(stddev=stddev)
    weights_init = tf.contrib.layers.xavier_initializer(uniform=False)
    # regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    regularizer = tf.contrib.layers.l1_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=prelu, normalizer_fn=slim.batch_norm):  # tf.keras.layers.PReLU
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc


def inference(images, bottleneck_layer_size=128, phase_train=False,
              weight_decay=0.00005, reuse=False, mobile=False):
    '''build a mobilenet_v2 graph to training or inference.

    Args:
        images: a tensor of shape [batch_size, height, width, channels].
        bottleneck_layer_size: number of predicted classes. If 0 or None, the logits layer
          is omitted and the input features to the logits layer (before dropout)
          are returned instead.
        phase_train: Whether or not we're training the model.
        weight_decay: The weight decay to use for regularizing the model.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        mobile: Using mobile or not.

    Returns:
        net: a 2D Tensor with the logits (pre-softmax activations) if bottleneck_layer_size
          is a non-zero integer, or the non-dropped-out input to the logits layer
          if bottleneck_layer_size is 0 or None.
        end_points: a dictionary from components of the network to the corresponding
          activation.

    Raises:
        ValueError: Input rank is invalid.
    '''
    arg_scope = mobilenet_v2_arg_scope(is_training=phase_train, weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        return mobilenet_v2(images, bottleneck_layer_size=bottleneck_layer_size,
                            is_training=phase_train, reuse=reuse, mobile=mobile)
