import numpy as np
import os

os.environ["KERAS_BACKEND"] = "torch"
from keras.layers import Layer
from keras import ops


class AdaINLayer(Layer) :
    def __init__(self) :
        super().__init__()

    def call(self, inputs) :
        content_input = inputs[0]
        style_input = inputs[1]

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
