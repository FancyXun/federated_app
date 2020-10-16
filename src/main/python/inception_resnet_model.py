import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np


class CenterLossLayer(keras.layers.Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(10, 2),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None):
        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)


        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True)
        return self.result  # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

# custom loss

def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)


def inception_resnet_stem(x):
    if K.image_data_format()== 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    c = layers.Conv2D(32, (3, 3), activation='relu', strides=(2, 2))(x)
    c = layers.Conv2D(32, (3, 3), activation='relu', )(c)
    c = layers.Conv2D(64, (3, 3), activation='relu', )(c)
    c = layers.MaxPooling2D((3, 3), strides=(2, 2))(c)
    c = layers.Conv2D(80, (1, 1), activation='relu', padding='same')(c)
    c = layers.Conv2D(192, (3, 3), activation='relu')(c)
    c = layers.Conv2D(256, (3, 3), activation='relu', strides=(2,2), padding='same')(c)
    b = layers.BatchNormalization(axis=channel_axis)(c)
    b = layers.Activation('relu')(b)
    return b

def inception_resnet_A(x, scale_residual=True):
    if K.image_data_format()== 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = x

    ir1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    ir2 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    ir2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(ir2)
    ir3 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    ir3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(ir3)
    ir3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(ir3)

    ir_merge = layers.concatenate([ir1, ir2, ir3], axis=channel_axis)

    ir_conv = layers.Conv2D(256, (1, 1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = layers.Lambda(lambda x: x * 0.1)(ir_conv)

    out = layers.add([init, ir_conv])
    out = layers.BatchNormalization(axis=channel_axis)(out)
    out = layers.Activation("relu")(out)
    return out

def inception_resnet_B(x, scale_residual=True):
    if K.image_data_format()== 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = x

    ir1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    ir2 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    ir2 = layers.Conv2D(128, (1, 7), activation='relu', padding='same')(ir2)
    ir2 = layers.Conv2D(128, (7, 1), activation='relu', padding='same')(ir2)

    ir_merge = layers.concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = layers.Conv2D(896, (1, 1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = layers.Lambda(lambda x: x * 0.1)(ir_conv)

    out = layers.add([init, ir_conv])
    out = layers.BatchNormalization(axis=channel_axis)(out)
    out = layers.Activation("relu")(out)
    return out

def inception_resnet_C(x, scale_residual=True):
    if K.image_data_format()== 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # x is relu activation
    init = x

    ir1 = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    ir2 = layers.Conv2D(192, (1, 1), activation='relu', padding='same')(x)
    ir2 = layers.Conv2D(192, (1, 3), activation='relu', padding='same')(ir2)
    ir2 = layers.Conv2D(192, (3, 1), activation='relu', padding='same')(ir2)

    ir_merge = layers.concatenate([ir1, ir2],axis=channel_axis)

    ir_conv = layers.Conv2D(1792, (1, 1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = layers.Lambda(lambda x: x * 0.1)(ir_conv)

    out = layers.add([init, ir_conv])
    out = layers.BatchNormalization(axis=channel_axis)(out)
    out = layers.Activation("relu")(out)
    return out

def reduction_A(x, k=192, l=224, m=256, n=384):
    if K.image_data_format()== 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = layers.MaxPooling2D((3,3), strides=(2,2))(x)

    r2 = layers.Conv2D(n, (3, 3), activation='relu', strides=(2,2))(x)
    r3 = layers.Conv2D(k, (1, 1), activation='relu', padding='same')(x)
    r3 = layers.Conv2D(l, (3, 3), activation='relu', padding='same')(r3)
    r3 = layers.Conv2D(m, (3, 3), activation='relu', strides=(2,2))(r3)

    m = layers.concatenate([r1, r2, r3], axis=channel_axis)
    m = layers.BatchNormalization(axis=channel_axis)(m)
    m = layers.Activation('relu')(m)
    return m


def reduction_resnet_B(x):
    if K.image_data_format()== 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = layers.MaxPooling2D((3,3), strides=(2,2), padding='valid')(x)

    r2 = layers.Conv2D(256, (1, 1), activation='relu', padding='same')(x)
    r2 = layers.Conv2D(384, (3, 3), activation='relu', strides=(2,2))(r2)
    r3 = layers.Conv2D(256, (1, 1), activation='relu', padding='same')(x)
    r3 = layers.Conv2D(256, (3, 3), activation='relu', strides=(2, 2))(r3)
    r4 = layers.Conv2D(256, (1, 1), activation='relu', padding='same')(x)
    r4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(r4)
    r4 = layers.Conv2D(256, (3, 3), activation='relu', strides=(2, 2))(r4)

    m = layers.concatenate([r1, r2, r3, r4], axis=channel_axis)
    m = layers.BatchNormalization(axis=channel_axis)(m)
    m = layers.Activation('relu')(m)
    return m

def create_inception_resnet_v1(nb_classes=10, scale=True):
    '''
    Creates a inception resnet v1 network

    :param nb_classes: number of classes.txt
    :param scale: flag to add scaling of activations
    :return: Keras Model with 1 input (299x299x3) input shape and 2 outputs (final_output, auxiliary_output)
    '''

    if K.image_data_format()== 'channels_first':
        init = layers.Input((3, 299, 299))
    else:
        init = layers.Input((299, 299, 3))
    input_label = layers.Input((nb_classes,))
    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_resnet_stem(init)

    # 5 x Inception Resnet A
    for i in range(5):
        x = inception_resnet_A(x, scale_residual=scale)

    # Reduction A - From Inception v4
    x = reduction_A(x, k=192, l=192, m=256, n=384)

    # 10 x Inception Resnet B
    for i in range(10):
        x = inception_resnet_B(x, scale_residual=scale)

    # # Auxiliary tower
    # aux_out = layers.AveragePooling2D((5, 5), strides=(3, 3))(x)
    # aux_out = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(aux_out)
    # aux_out = layers.Conv2D(768, (5, 5), activation='relu')(aux_out)
    # aux_out = layers.Flatten()(aux_out)
    # aux_out = layers.Dense(nb_classes, activation='softmax')(aux_out)

    # Reduction Resnet B
    x = reduction_resnet_B(x)

    # 5 x Inception Resnet C
    for i in range(5):
        x = inception_resnet_C(x, scale_residual=scale)

    # Average Pooling
    x = layers.AveragePooling2D((8,8))(x)

    # Dropout
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)

    # Output
    out = layers.Dense(nb_classes, activation='softmax', name='main_out')(x)
    side_out = CenterLossLayer(alpha=0.5, name='centerlosslayer')([x, input_label])

    model = keras.Model(inputs=[init, input_label], outputs=[out, side_out], name='Inception-Resnet-v1')

    return model

def get_inception_resnet_centerloss_model(nb_classes):
    lambda_centerloss = 0.1
    initial_learning_rate = 1e-3
    model = create_inception_resnet_v1(nb_classes)
    optim = keras.optimizers.SGD(lr=initial_learning_rate, momentum=0.9)
    model.compile(optimizer=optim,
                  loss=[keras.losses.categorical_crossentropy, zero_loss],
                  loss_weights=[1, lambda_centerloss],
                  metrics={'main_out': tf.keras.metrics.AUC()})
    return model
