import tensorflow as tf
import numpy as np

conf_threshold = 0.6
nms_iou_threshold = 0.3
nms_max_output_size = 200
top_k = 100
center_variance = 0.1
size_variance = 0.2

image_size = [320, 240]  # default input size 320*240
feature_map_wh_list = [[40, 30], [20, 15], [10, 8], [5, 4]]  # default feature map size
min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]


def basic_conv(x, out_ch, kernel_size, stride=(1, 1), padding=0, dilation=1, relu=True,
               bn=True, prefix='basic_conv'):
    if 0 < padding:
        out = tf.keras.layers.ZeroPadding2D(padding=padding, name=f'{prefix}_padding')(x)
    else:
        out = x
    out = tf.keras.layers.Conv2D(out_ch,
                                 kernel_size,
                                 strides=stride,
                                 dilation_rate=dilation,
                                 use_bias=(not bn),
                                 name=f'{prefix}_conv')(out)
    if bn:
        out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}_bn')(out)
    if relu:
        out = tf.keras.layers.ReLU(name=f'{prefix}_relu')(out)

    return out


def conv_bn(x, out_ch, stride, padding=1, prefix='conv_bn'):
    out = tf.keras.layers.ZeroPadding2D(padding=padding, name=f'{prefix}.0_padding')(x)
    out = tf.keras.layers.Conv2D(out_ch,
                                 (3, 3),
                                 strides=stride,
                                 use_bias=False,
                                 name=f'{prefix}.0_conv')(out)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}.1_bn')(out)
    out = tf.keras.layers.ReLU(name=f'{prefix}.2_relu')(out)

    return out


def conv_dw(x, out_ch, stride, padding=1, prefix='conv_dw'):
    out = tf.keras.layers.ZeroPadding2D(padding=padding, name=f'{prefix}.0_padding')(x)
    out = tf.keras.layers.DepthwiseConv2D(3, strides=stride,
                                          use_bias=False,
                                          name=f'{prefix}.0_dconv')(out)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}.1_bn')(out)
    out = tf.keras.layers.ReLU(name=f'{prefix}.2_relu')(out)

    out = tf.keras.layers.Conv2D(out_ch, 1, use_bias=False, name=f'{prefix}.3_conv')(out)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}.4_bn')(out)
    out = tf.keras.layers.ReLU(name=f'{prefix}.5_relu')(out)

    return out


def basic_rfb(x, in_ch, out_ch, stride=1, scale=0.1, map_reduce=8, vision=1, prefix='basic_rfb'):
    inter_ch = in_ch // map_reduce

    branch0 = basic_conv(x, inter_ch, kernel_size=1, stride=1, relu=False,
                         prefix=f'{prefix}.branch0.0')
    branch0 = basic_conv(branch0, 2 * inter_ch, kernel_size=3, stride=stride, padding=1,
                         prefix=f'{prefix}.branch0.1')
    branch0 = basic_conv(branch0, 2 * inter_ch, kernel_size=3, stride=1, dilation=vision + 1,
                         padding=vision + 1, relu=False, prefix=f'{prefix}.branch0.2')

    branch1 = basic_conv(x, inter_ch, kernel_size=1, stride=1, relu=False,
                         prefix=f'{prefix}.branch1.0')
    branch1 = basic_conv(branch1, 2 * inter_ch, kernel_size=3, stride=stride, padding=1,
                         prefix=f'{prefix}.branch1.1')
    branch1 = basic_conv(branch1, 2 * inter_ch, kernel_size=3, stride=1, dilation=vision + 2,
                         padding=vision + 2, relu=False, prefix=f'{prefix}.branch1.2')

    branch2 = basic_conv(x, inter_ch, kernel_size=1, stride=1, relu=False,
                         prefix=f'{prefix}.branch2.0')
    branch2 = basic_conv(branch2, (inter_ch // 2) * 3, kernel_size=3, stride=1, padding=1,
                         prefix=f'{prefix}.branch2.1')
    branch2 = basic_conv(branch2, 2 * inter_ch, kernel_size=3, stride=stride, padding=1,
                         prefix=f'{prefix}.branch2.2')
    branch2 = basic_conv(branch2, 2 * inter_ch, kernel_size=3, stride=1, dilation=vision + 4,
                         padding=vision + 4, relu=False, prefix=f'{prefix}.branch2.3')

    out = tf.keras.layers.Concatenate(axis=-1, name=f'{prefix}_cat')([branch0, branch1, branch2])
    out = basic_conv(out, out_ch, kernel_size=1, stride=1, relu=False, prefix=f'{prefix}.convlinear')
    shortcut = basic_conv(x, out_ch, kernel_size=1, stride=stride, relu=False, prefix=f'{prefix}.shortcut')
    out = tf.multiply(out, scale, name=f'{prefix}_mul')
    out = tf.keras.layers.Add(name=f'{prefix}_add')([out, shortcut])
    out = tf.keras.layers.ReLU(name=f'{prefix}_relu')(out)

    return out


def separable_conv(x, out_ch, kernel_size, stride, padding, prefix='separable_conv'):
    out = tf.keras.layers.ZeroPadding2D(padding=padding, name=f'{prefix}_dconv_padding')(x)

    out = tf.keras.layers.DepthwiseConv2D(kernel_size,
                                          strides=stride,
                                          name=f'{prefix}_dconvbias')(out)
    out = tf.keras.layers.ReLU(name=f'{prefix}_relu')(out)
    out = tf.keras.layers.Conv2D(out_ch, 1,
                                 name=f'{prefix}_convbias')(out)

    return out


def decode_regression(reg, image_size, feature_map_w_h_list, min_boxes,
                      center_variance, size_variance):
    priors = []
    for feature_map_w_h, min_box in zip(feature_map_w_h_list, min_boxes):
        xy_grid = np.meshgrid(range(feature_map_w_h[0]), range(feature_map_w_h[1]))
        xy_grid = np.add(xy_grid, 0.5)
        xy_grid[0, :, :] /= feature_map_w_h[0]
        xy_grid[1, :, :] /= feature_map_w_h[1]
        xy_grid = np.stack(xy_grid, axis=-1)
        xy_grid = np.tile(xy_grid, [1, 1, len(min_box)])
        xy_grid = np.reshape(xy_grid, (-1, 2))

        wh_grid = np.array(min_box) / np.array(image_size)[:, np.newaxis]
        wh_grid = np.tile(np.transpose(wh_grid), [np.product(feature_map_w_h), 1])

        prior = np.concatenate((xy_grid, wh_grid), axis=-1)
        priors.append(prior)

    priors = np.concatenate(priors, axis=0)
    print(f'priors nums:{priors.shape[0]}')

    priors = tf.constant(priors, dtype=tf.float32, shape=priors.shape, name='priors')

    center_xy = reg[..., :2] * center_variance * priors[..., 2:] + priors[..., :2]
    center_wh = tf.exp(reg[..., 2:] * size_variance) * priors[..., 2:]

    # center to corner
    start_xy = center_xy - center_wh / 2
    end_xy = center_xy + center_wh / 2

    loc = tf.concat([start_xy, end_xy], axis=-1)
    loc = tf.clip_by_value(loc, clip_value_min=0.0, clip_value_max=1.0)

    return loc


def post_processing(reg_list, cls_list, num_classes, image_size, feature_map_wh_list, min_boxes,
                    center_variance, size_variance,
                    conf_threshold=0.6, nms_max_output_size=100, nms_iou_threshold=0.3, top_k=100):
    reg_list = [tf.keras.layers.Reshape([-1, 4])(reg) for reg in reg_list]
    cls_list = [tf.keras.layers.Reshape([-1, num_classes])(cls) for cls in cls_list]

    reg = tf.keras.layers.Concatenate(axis=1)(reg_list)
    cls = tf.keras.layers.Concatenate(axis=1)(cls_list)

    # post process
    cls = tf.keras.layers.Softmax(axis=-1)(cls)
    loc = decode_regression(reg, image_size, feature_map_wh_list, min_boxes,
                            center_variance, size_variance)

    result = tf.keras.layers.Concatenate(axis=-1)([cls, loc])

    # confidence thresholding
    mask = conf_threshold < cls[..., 1]
    result = tf.boolean_mask(tensor=result, mask=mask)

    # non-maximum suppression
    mask = tf.image.non_max_suppression(boxes=result[..., -4:],
                                        scores=result[..., 1],
                                        max_output_size=nms_max_output_size,
                                        iou_threshold=nms_iou_threshold,
                                        name='non_maximum_suppresion')
    result = tf.gather(params=result, indices=mask, axis=0)

    # top-k filtering
    top_k_value = tf.math.minimum(tf.constant(top_k), tf.shape(result)[0])
    mask = tf.nn.top_k(result[..., 1], k=top_k_value, sorted=True).indices
    result = tf.gather(params=result, indices=mask, axis=0)

    return result

def create_rfb_net(input_shape, base_channel, num_classes):
    input_node = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 3))

    net = conv_bn(input_node, base_channel, stride=2, prefix='basenet.0')  # 120x160
    net = conv_dw(net, base_channel * 2, stride=1, prefix='basenet.1')
    net = conv_dw(net, base_channel * 2, stride=2, prefix='basenet.2')  # 60x80
    net = conv_dw(net, base_channel * 2, stride=1, prefix='basenet.3')
    net = conv_dw(net, base_channel * 4, stride=2, prefix='basenet.4')  # 30x40
    net = conv_dw(net, base_channel * 4, stride=1, prefix='basenet.5')
    net = conv_dw(net, base_channel * 4, stride=1, prefix='basenet.6')
    header_0 = basic_rfb(net, base_channel * 4, base_channel * 4, stride=1, scale=1.0, prefix='basenet.7')
    net = conv_dw(header_0, base_channel * 8, stride=2, prefix='basenet.8')  # 15x20
    net = conv_dw(net, base_channel * 8, stride=1, prefix='basenet.9')
    header_1 = conv_dw(net, base_channel * 8, stride=1, prefix='basenet.10')
    net = conv_dw(header_1, base_channel * 16, stride=2, prefix='basenet.11')  # 8x10
    header_2 = conv_dw(net, base_channel * 16, stride=1, prefix='basenet.12')

    out = tf.keras.layers.Conv2D(base_channel * 4, 1, padding='SAME', name='extras_convbias')(header_2)
    out = tf.keras.layers.ReLU(name='extras_relu1')(out)
    out = separable_conv(out, base_channel * 16, kernel_size=3, stride=2, padding=1,
                         prefix='extras_sep')
    header_3 = tf.keras.layers.ReLU(name='extras_relu2')(out)

    reg_0 = separable_conv(header_0, 3 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_0_sep')
    cls_0 = separable_conv(header_0, 3 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_0_sep')

    reg_1 = separable_conv(header_1, 2 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_1_sep')
    cls_1 = separable_conv(header_1, 2 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_1_sep')

    reg_2 = separable_conv(header_2, 2 * 4, kernel_size=3, stride=1, padding=1,
                           prefix='reg_2_sep')
    cls_2 = separable_conv(header_2, 2 * num_classes, kernel_size=3, stride=1, padding=1,
                           prefix='cls_2_sep')

    reg_3 = tf.keras.layers.Conv2D(3 * 4, kernel_size=3, padding='SAME',
                                   name='reg_3_convbias')(header_3)
    cls_3 = tf.keras.layers.Conv2D(3 * num_classes, kernel_size=3, padding='SAME',
                                   name='cls_3_convbias')(header_3)

    result = post_processing([reg_0, reg_1, reg_2, reg_3],
                             [cls_0, cls_1, cls_2, cls_3],
                             num_classes, image_size, feature_map_wh_list, min_boxes,
                             center_variance, size_variance)

    model = tf.keras.Model(inputs=[input_node], outputs=[result])
    model.summary()

    return model


input_shape = (240, 320)  # H,W
base_channel = 8 * 2
num_classes = 2
model = create_rfb_net(input_shape, base_channel, num_classes)