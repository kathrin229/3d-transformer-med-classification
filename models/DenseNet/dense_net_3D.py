"""
Creating a 3D DenseNet 121
Code modified from https://towardsdatascience.com/creating-densenet-121-with-tensorflow-edbc08a956d8
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool3D, GlobalAveragePooling3D, MaxPool3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K

def densenet3D(input_shape, n_classes, filters=32):
    """
        Creates a tf/keras 3D Densenet121 model
        Args:
            input_shape (tuple): the input shape - width x height x depth x channels
                In our case (80, 64, 16, 1) for the smaller datasets and (160, 128, 32, 1) for the bigger dataset.
            n_classes (int): The number of classes to be predicted.
            filters (int): number of filters, optional
        Returns:
            model (tf.keras.models.Model): The 3D Densenet121 model
    """
    def bn_rl_conv(x, filters, kernel=1, strides=1):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(filters, kernel, strides=strides, padding='same')(x)
        return x
    
    def dense_block(x, repetition):
        for _ in range(repetition):
            y = bn_rl_conv(x, 4*filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y,x])
        return x
        
    def transition_layer(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
        x = AvgPool3D(2, strides = 2, padding = 'same')(x)
        return x
    
    input = Input (input_shape)
    x = Conv3D(64, 7, strides = 2, padding = 'same')(input)
    x = MaxPool3D(3, strides = 2, padding = 'same')(x)
    
    for repetition in [6,12,24,16]:
        d = dense_block(x, repetition)
        x = transition_layer(d)

    x = GlobalAveragePooling3D()(d)
    output = Dense(n_classes, activation = 'softmax')(x)
    
    model = Model(input, output)
    return model
