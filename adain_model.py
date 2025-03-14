import numpy as np
import torch

import os
import shutil
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.layers import Layer # type: ignore
from keras import ops, Model, Loss
import mlflow
from pathlib import Path
import json

def instance_mean(batch) :
    shape = batch.shape
    match len(shape) :
        case 4 :
            pass
        case 3 :
            batch = batch.unsqueeze(0)
        case _ :
            raise ValueError("Incorrect shape for the provided batch. Expected a 3- or 4-dimensional Tensor.")
    
    return batch.mean(axis=[1,2], keepdims=True)

def instance_std(batch) :
    shape = batch.shape
    match len(shape) :
        case 4 :
            pass
        case 3 :
            batch = batch.unsqueeze(0)
        case _ :
            raise ValueError("Incorrect shape for the provided batch. Expected a 3- or 4-dimensional Tensor.")
    
    return batch.std(axis=[1,2], keepdims=True) + 1e-6 # Constant added for numerical stability


@keras.saving.register_keras_serializable()
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

    def call(self, inputs, all_features=False) :
        relu1_features = self.relu1_1(inputs)
        relu2_features = self.relu2_1(relu1_features)
        relu3_features = self.relu3_1(relu2_features)
        relu4_features = self.relu4_1(relu3_features)

        if all_features :
            return relu1_features, relu2_features, relu3_features, relu4_features
        else :
            return relu4_features # During inference, only the final features are needed
    

@keras.saving.register_keras_serializable()
class AdaINLayer(Layer) :
    def __init__(self) :
        super().__init__()

    def call(self, content_input, style_input) :
       return instance_std(style_input) * ((content_input - instance_mean(content_input)) / instance_std(content_input)) + instance_mean(style_input)
    
    def get_config(self):
        config = super().get_config()
        return config
    
    def build(self, input_shape) :
        super().build(input_shape)    


@keras.saving.register_keras_serializable()
class ReflectiveConv2D(keras.layers.Conv2D) :
    """Custom Conv2D layer with reflective padding"""
    def __init__(self, filters, kernel_size, pad_width = ((0,0), (1,1), (1,1), (0,0)), strides=(1,1), data_format="channels_last", dilation_rate=(1,1), groups=1, activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        self.pad_width = pad_width
        super().__init__(filters, kernel_size, strides, padding="valid", data_format=data_format, dilation_rate=dilation_rate, groups=groups, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
    
    def call(self, inputs):
        inputs = keras.ops.pad(inputs, pad_width=self.pad_width, mode="reflect")
        return super().call(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({"pad_width": self.pad_width})
        return config
    
    def build(self, input_shape) :
        super().build(input_shape)


@keras.saving.register_keras_serializable()
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
            ReflectiveConv2D(3, (3,3), activation=None)
        ), trainable=True)
    
    def call(self, inputs) :
        return self.decoder(inputs)
    
    def build(self, input_shape) :
        super().build(input_shape)


@keras.saving.register_keras_serializable()
class AdaINModel(Model) :
    def __init__(self, lamb = 1.0, loss_reduction = "sum_over_batch_size", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

        self.encoder = VGGEncoder().to(self.device)
        self.ada_in = AdaINLayer().to(self.device)
        self.decoder = VGGDecoder().to(self.device)

        self.loss_ = AdaINLoss(lamb=lamb, reduction=loss_reduction)

        self.input_shape = (None, None, None, 3)
        self.output_shape = (None, None, None, 3)
        self.build(input_shape=[(None, 256, 256, 3), (None, 256, 256, 3)])
    
    @staticmethod
    def load_from_mlflow(run_id, mlflow_tracking_uri) :
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        tmp_path = Path('./_tmp_download')
        arch_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="architecture.json", dst_path=tmp_path / 'architecture.json'))
        weights_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="weights.weights.h5", dst_path=tmp_path / 'weights.weights.h5'))

        with open(arch_path, 'r') as f :
            json_config = json.load(f)

        model = keras.models.model_from_json(json.dumps(json_config))

        # Build the model as expected
        input = [torch.zeros((1,256,256,3)), torch.zeros((1,256,256,3))]
        model.compile(optimizer = keras.optimizers.get(json_config["compile_config"]["optimizer"]["config"]["name"]))
        model.train_on_batch(input) # This step is required for Keras to build the model layers and the optimizer variables
        model.load_weights(weights_path)

        # Remove downloaded files once used to prevent useless disk storage
        shutil.rmtree(tmp_path)

        return model

    def build(self, input_shape) :
        super().build(input_shape)

    def call(self, inputs, training=False):
        content = inputs[0].to(self.device)
        content = keras.applications.vgg19.preprocess_input(content) # Make sure the format of input is compatible with VGG (BGR, mean-centered)
        style = inputs[1].to(self.device)
        style = keras.applications.vgg19.preprocess_input(style)
        resized_style = keras.layers.Resizing(content.shape[1], content.shape[2])(style)

        encoded_content = self.encoder(content) 
        style_full_features = self.encoder(resized_style, all_features=True)
        encoded_style = style_full_features[-1]

        combined_features = self.ada_in(encoded_content, encoded_style)

        decoded_img = self.decoder(combined_features)
        gen_image_full_features = self.encoder(decoded_img, all_features=True)

        self.add_loss(self.loss_.total_loss(gen_image_full_features, style_full_features, combined_features))
        
        return decoded_img


class AdaINLoss() :
    def __init__(self, lamb, reduction="sum_over_batch_size") :
        """The custom loss function defined in the AdaIN paper

        Args:
            lamb (float): Relative weighing of the style loss wrt the content loss
        """
        self.lamb = lamb
        self.reduction = reduction

    def batched_euclidean_distance(self, tensor1, tensor2) :
        """Compute the pairwise euclidean distances between two batches of Tensors

        Args:
            tensor1 (tensor): First tensor batch
            tensor2 (tensor): Second tensor batch

        Returns:
            tensor: The batched euvclidean distance
        """
        return torch.sqrt(torch.sum((tensor1 - tensor2)**2, dim=(1,2,3)))

    def content_loss(self, generated_features, adain_output) :
        """Compute the euclidean distance between the features of the generated image and the output of the AdaIN layer

        Args:
            generated_features (Tensor): Result of the encoding of the generated image through the VGG19-encoder
            adain_output (Tensor): Output of the AdaIN layer, i.e features of the content image renormalized with style statistics

        Returns:
            Tensor: The euclidean distance (or 2-norm) between the two tensors
        """
        return self.batched_euclidean_distance(generated_features, adain_output)

    def style_loss(self, generated_full_features, style_full_features) :
        """Compute the style loss by computing euclidean distances between basis statistics from features of the generated image and the style image

        Args:
            generated_full_features (Tensor): Result of the encoding of the generated image (Computed for each first ReLU of VGG blocks)
            style_full_features (Tensor): Result of the encoding of the style image (Computed for each first ReLU of VGG blocks)

        Returns:
            Tensor: The accumulation of euclidean distances between statistics
        """
        if not isinstance(generated_full_features, tuple) :
            generated_full_features = (generated_full_features)

        style_loss = torch.zeros(generated_full_features[0].shape[0]).to(generated_full_features[0].device)
        for layer in range(len(generated_full_features)) :
            gen_mean = instance_mean(generated_full_features[layer])
            gen_std = instance_std(generated_full_features[layer])
            style_mean = instance_mean(style_full_features[layer])
            style_std = instance_std(style_full_features[layer])
            
            style_loss += self.batched_euclidean_distance(gen_mean, style_mean) + self.batched_euclidean_distance(gen_std, style_std)
    
        return style_loss
    
    def total_loss(self, generated_full_features, style_full_features, adain_output) :
        generated_features = generated_full_features[-1]
        batched_loss = self.content_loss(generated_features, adain_output) + self.lamb * self.style_loss(generated_full_features, style_full_features)

        match self.reduction :
            case "sum_over_batch_size" | "mean" :
                return batched_loss.mean()
            case "sum" :
                return batched_loss.sum()
            case _ :
                return batched_loss