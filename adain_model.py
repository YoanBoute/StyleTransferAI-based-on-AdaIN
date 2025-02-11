import numpy as np
import torch

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.layers import Layer # type: ignore
from keras import ops, Model, Loss


def instance_mean(batch) :
    shape = batch.shape
    match len(shape) :
        case 4 :
            pass
        case 3 :
            batch = batch.unsqueeze(0)
        case _ :
            raise ValueError("Incorrect shape for the provided batch. Expected a 3- or 4-dimensional Tensor.")
    
    return ops.mean(batch, [2,3], keepdims=True)

def instance_std(batch) :
    shape = batch.shape
    match len(shape) :
        case 4 :
            pass
        case 3 :
            batch = batch.unsqueeze(0)
        case _ :
            raise ValueError("Incorrect shape for the provided batch. Expected a 3- or 4-dimensional Tensor.")
    
    return ops.std(batch, [2,3], keepdims=True)


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

        # The network is subdivided to have access to the features of different depths (for the style loss computation especially)
        self.relu1_1 = keras.Sequential(basis_vgg.layers[:2])
        self.relu2_1 = keras.Sequential(basis_vgg.layers[2:5])
        self.relu3_1 = keras.Sequential(basis_vgg.layers[5:8])        
        self.relu4_1 = keras.Sequential(basis_vgg.layers[8:13]) # As described in the AdaIN paper, the encoder is the VGG19 network up to the relu4_1 layer
        
        self.relu1_1.trainable = False
        self.relu2_1.trainable = False
        self.relu3_1.trainable = False
        self.relu4_1.trainable = False

    def call(self, inputs, training=False) :
        relu1_features = self.relu1_1(inputs)
        relu2_features = self.relu2_1(relu1_features)
        relu3_features = self.relu3_1(relu2_features)
        relu4_features = self.relu4_1(relu3_features)

        if training :
            return relu1_features, relu2_features, relu3_features, relu4_features
        else :
            return relu4_features # During inference, only the final features are needed
    

class AdaINLayer(Layer) :
    def __init__(self) :
        super().__init__()
        self.trainable = False

    def call(self, content_input, style_input) :
        # content_input = inputs[0]
        # style_input = inputs[1]

        return instance_std(style_input) * ((content_input - instance_mean(content_input)) / instance_std(content_input)) + instance_mean(style_input)


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
        
        if training :
            encoded_content = self.encoder(content, training=True)[-1] 
            style_full_features = self.encoder(resized_style, training=True)
            encoded_style = style_full_features[-1]
        else :
            encoded_content = self.encoder(content)
            encoded_style = self.encoder(resized_style)

        combined_features = self.ada_in(encoded_content, encoded_style)
        decoded_img = self.decoder(combined_features)
        
        if training :
            gen_image_full_features = self.encoder(decoded_img, training=True)
            return gen_image_full_features, style_full_features, combined_features # During training, the generated image is not important, but its features are for the loss computation
        else :
            return decoded_img 


class AdaINLoss(Loss) :
    def __init__(self, lamb, name=None, reduction="sum_over_batch_size", dtype=None):
        """The custom loss function defined in the AdaIN paper

        Args:
            lamb (float): Relative weighing of the style loss wrt the content loss
        """
        super().__init__(name, reduction, dtype)
        self.lamb = lamb
    
    def content_loss(self, generated_features, adain_output) :
        """Compute the euclidean distance between the features of the generated image and the output of the AdaIN layer

        Args:
            generated_features (Tensor): Result of the encoding of the generated image through the VGG19-encoder
            adain_output (Tensor): Output of the AdaIN layer, i.e features of the content image renormalized with style statistics

        Returns:
            Tensor: The euclidean distance (or 2-norm) between the two tensors
        """
        return torch.cdist(generated_features, adain_output, p=2.0)

    def style_loss(self, generated_full_features, style_full_features) :
        """Compute the style loss by computing euclidean distances between basis statistics from features of the generated image and the style image

        Args:
            generated_full_features (Tensor): Result of the encoding of the generated image (Computed for each first ReLU of VGG blocks)
            style_full_features (Tensor): Result of the encoding of the style image (Computed for each first ReLU of VGG blocks)

        Returns:
            Tensor: The accumulation of euclidean distances between statistics
        """
        style_loss = 0
        for layer in enumerate(generated_full_features) :
            gen_mean = instance_mean(generated_full_features[layer])
            gen_std = instance_std(generated_full_features[layer])
            style_mean = instance_mean(style_full_features[layer])
            style_std = instance_std(style_full_features[layer])

            style_loss += torch.cdist(gen_mean, style_mean, p=2.0) + torch.cdist(gen_std, style_std, p=2.0)
        
        return style_loss
    
    def call(self, y_pred, y_true) :
        (generated_full_features, style_full_features, adain_output) = y_pred
        generated_features = generated_full_features[-1]
        return self.content_loss(generated_features, adain_output) + self.lamb * self.style_loss(generated_full_features, style_full_features)