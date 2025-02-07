import numpy as np

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.layers import Layer
from keras import ops, Model


class VGGEncoder(Layer) :
    def __init__(self) :
        super().__init__()
        basis_vgg = keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            name="vgg19"
        )

        # As described in the AdaIN paper, the encoder is the VGG19 network up to the relu4_1 layer        
        self.encoder = keras.Sequential(basis_vgg.layers[:13])
        self.encoder.trainable = False

    def call(self, inputs) :
        return self.encoder(inputs)
    

class AdaINLayer(Layer) :
    def __init__(self) :
        super().__init__()
        self.trainable = False

    def call(self, content_input, style_input) :
        # content_input = inputs[0]
        # style_input = inputs[1]

        return self.instance_std(style_input) * ((content_input - self.instance_mean(content_input)) / self.instance_std(content_input)) + self.instance_mean(style_input)


    def instance_mean(self, batch) :
        shape = batch.shape
        match len(shape) :
            case 4 :
                pass
            case 3 :
                batch = batch.unsqueeze(0)
            case _ :
                raise ValueError("Incorrect shape for the provided batch. Expected a 3- or 4-dimensional Tensor.")
        
        return ops.mean(batch, [2,3], keepdims=True)

    def instance_std(self, batch) :
        shape = batch.shape
        match len(shape) :
            case 4 :
                pass
            case 3 :
                batch = batch.unsqueeze(0)
            case _ :
                raise ValueError("Incorrect shape for the provided batch. Expected a 3- or 4-dimensional Tensor.")
        
        return ops.std(batch, [2,3], keepdims=True)


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
    

class AdaINModel(Model) :
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = VGGEncoder()
        self.ada_in = AdaINLayer()
        self.decoder = VGGDecoder()
    
    def call(self, inputs, training=False):
        content = inputs[0]
        style = inputs[1]
        resized_style = keras.layers.Resizing(content.shape[1], content.shape[2])(style) # To be combined in AdaIN layer, content and style image must have the same initial size
        encoded_content = self.encoder(content)
        encoded_style = self.encoder(resized_style)
        combined_features = self.ada_in(encoded_content, encoded_style)
        decoded_img = self.decoder(combined_features)
        if training :
            return decoded_img, encoded_style, combined_features # During training, the result of the style encoding and of the AdaIN layer are necessary for the loss computation
        else :
            return decoded_img 