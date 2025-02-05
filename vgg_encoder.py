import numpy as np
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.layers import Layer

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
                
        self.encoder = keras.Sequential(basis_vgg.layers[:13])

    def build(self, input_shape) :
        pass

    def call(self, inputs) :
        return self.encoder(inputs)
