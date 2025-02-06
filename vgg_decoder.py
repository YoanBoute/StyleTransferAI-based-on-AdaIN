import numpy as np
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.layers import Layer

class ReflectiveConv2D(keras.layers.Conv2D) :
    """Custom Conv2D layer with reflective padding"""
    def __init__(self, filters, kernel_size, pad_width = ((0,0), (1,1), (1,1), (0,0)), strides=(1,1), data_format="channels_last", dilation_rate=(1,1), groups=1, activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        self.pad_width = pad_width
        super().__init__(filters, kernel_size, strides, padding="valid", data_format=data_format, dilation_rate=dilation_rate, groups=groups, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
    
    def call(self, inputs):
        inputs = keras.ops.pad(inputs, pad_width=self.pad_width, mode="reflect")
        return super().call(inputs)

class VGGDecoder(Layer) :
    """Decoder described in the AdaIN paper, built symmetrically to the encoder, using reflective padding and nearest up-sampling"""
    def __init__(self) :
        super().__init__()
        self.decoder = keras.Sequential((
            ReflectiveConv2D(256, (3,3), activation='relu'),
            keras.layers.UpSampling2D(size=(2,2), data_format="channels_last", interpolation="nearest"),
            ReflectiveConv2D(256, (3,3), activation='relu'),
            ReflectiveConv2D(256, (3,3), activation='relu'),
            ReflectiveConv2D(256, (3,3), activation='relu'),
            ReflectiveConv2D(128, (3,3), activation='relu'),
            keras.layers.UpSampling2D(size=(2,2), data_format="channels_last", interpolation="nearest"),
            ReflectiveConv2D(128, (3,3), activation='relu'),
            ReflectiveConv2D(64, (3,3), activation='relu'),
            keras.layers.UpSampling2D(size=(2,2), data_format="channels_last", interpolation="nearest"),
            ReflectiveConv2D(64, (3,3), activation='relu'),
            ReflectiveConv2D(3, (3,3), activation='relu')
        ), trainable=True)
    
    def call(self, inputs) :
        return self.decoder(inputs)