import numpy as np
import torch

import os
import shutil
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.layers import Layer, Input # type: ignore
from keras import ops, Model, Loss
import mlflow
from pathlib import Path
import json
import matplotlib.pyplot as plt
import gc

def instance_mean(batch):
    return ops.mean(batch, axis=[1, 2], keepdims=True)

def instance_std(batch):
    return ops.std(batch, axis=[1, 2], keepdims=True) + 1e-6 # Constant added for numerical stability


@keras.saving.register_keras_serializable()
class ReflectiveConv2D(keras.layers.Conv2D):
    """Custom Conv2D layer with reflective-same padding"""
    def __init__(self, filters, kernel_size, strides=(1,1), data_format="channels_last", dilation_rate=(1,1), groups=1, activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        super().__init__(filters, kernel_size, strides=strides, padding="valid", data_format=data_format, dilation_rate=dilation_rate, groups=groups, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
    
    def call(self, inputs):
        pad_h = self.kernel_size[0] // 2 # Dynamic padding computation
        pad_w = self.kernel_size[1] // 2  
        pad_width = ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0))  
        inputs = keras.ops.pad(inputs, pad_width=pad_width, mode="reflect")
        return super().call(inputs)
    

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

        # Replace all conv layers with ReflectivePadding conv layers
        inputs = Input(shape=(None, None, 3))
        x = inputs
        for layer in basis_vgg.layers :
            if isinstance(layer, keras.layers.InputLayer) :
                x = layer.output
            elif isinstance(layer, keras.layers.Conv2D) :
                x = ReflectiveConv2D(filters=layer.filters, kernel_size=layer.kernel_size, strides=layer.strides, activation=layer.activation, use_bias=layer.use_bias, name=layer.name)(x)
            else :
                x = layer(x)
        reflective_vgg = Model(inputs, x)

        layers_names = [l.name for l in reflective_vgg.layers]
        for layer in basis_vgg.layers:
            if layer.name in layers_names :
                reflective_vgg.get_layer(layer.name).set_weights(layer.get_weights())

        # The network is subdivided to have access to the features of different depths (for the style loss computation especially)
        self.relu1_1 = keras.Sequential(reflective_vgg.layers[:2])
        self.relu2_1 = keras.Sequential(reflective_vgg.layers[2:5])
        self.relu3_1 = keras.Sequential(reflective_vgg.layers[5:8])        
        self.relu4_1 = keras.Sequential(reflective_vgg.layers[8:13]) # As described in the AdaIN paper, the encoder is the VGG19 network up to the relu4_1 layer
        
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

class DepthToSpace(Layer):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def call(self, input):
        batch, height, width, depth = ops.shape(input)
        depth = depth // (self.block_size**2)

        x = ops.reshape(
            input, [batch, height, width, self.block_size, self.block_size, depth]
        )
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(
            x, [batch, height * self.block_size, width * self.block_size, depth]
        )
        return x

@keras.saving.register_keras_serializable()
class VGGDecoder(Layer) :
    """Decoder described in the AdaIN paper, built symmetrically to the encoder, using reflective padding and nearest up-sampling"""
    def __init__(self) :
        super().__init__()
        self.decoder = keras.Sequential((
            # ReflectiveConv2D(256, (3,3), activation='relu'),
            # keras.layers.UpSampling2D(size=(2,2), data_format="channels_last", interpolation="bilinear"),
            # ReflectiveConv2D(256, (3,3), activation='relu'),
            # ReflectiveConv2D(256, (3,3), activation='relu'),
            # ReflectiveConv2D(256, (3,3), activation='relu'),
            # ReflectiveConv2D(128, (3,3), activation='relu'),
            # keras.layers.UpSampling2D(size=(2,2), data_format="channels_last", interpolation="bilinear"),
            # ReflectiveConv2D(128, (3,3), activation='relu'),
            # ReflectiveConv2D(64, (3,3), activation='relu'),
            # keras.layers.UpSampling2D(size=(2,2), data_format="channels_last", interpolation="bilinear"),
            # ReflectiveConv2D(64, (3,3), activation='relu'),
            # ReflectiveConv2D(3, (3,3), activation=None)

            ReflectiveConv2D(256, (3,3), activation='relu'),
            DepthToSpace(2),
            ReflectiveConv2D(64, (3,3), activation='relu'),
            ReflectiveConv2D(64, (3,3), activation='relu'),
            ReflectiveConv2D(64, (3,3), activation='relu'),
            DepthToSpace(2),
            ReflectiveConv2D(16, (3,3), activation='relu'),
            ReflectiveConv2D(16, (3,3), activation='relu'),
            DepthToSpace(2),
            ReflectiveConv2D(4, (3,3), activation='relu'),
            ReflectiveConv2D(3, (3,3), activation=None)
        ), trainable=True)
    
    def call(self, inputs) :
        return self.decoder(inputs)
    
    def build(self, input_shape) :
        super().build(input_shape)


@keras.saving.register_keras_serializable()
class AdaINModel(Model) :
    def __init__(self, style_loss_weight = 1.0, tv_loss_weight = 0.1, loss_reduction = "sum_over_batch_size", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

        self.encoder = VGGEncoder().to(self.device)
        self.ada_in = AdaINLayer().to(self.device)
        self.decoder = VGGDecoder().to(self.device)

        self.loss_ = AdaINLoss(style_loss_weight=style_loss_weight, tv_weight=tv_loss_weight, reduction=loss_reduction)

        self.input_shape = (None, None, None, 3)
        self.output_shape = (None, None, None, 3)
        self.build(input_shape=[(None, 256, 256, 3), (None, 256, 256, 3)])
    
    @staticmethod
    def load_from_mlflow(run_id, mlflow_tracking_uri) :
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        tmp_path = Path('./_tmp_download')
        try : 
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

            # with h5py.File(weights_path, 'r') as f:
            #     block_name = 'relu1_1'
            #     layer_name = 'conv2d'
            #     model.encoder.relu1_1.layers[0].set_weights([f["encoder"][block_name]["layers"][layer_name]["vars"]['0'], f["encoder"][block_name]["layers"][layer_name]["vars"]['1']])
            #     block_name = 'relu2_1'
            #     layer_name = 'conv2d'
            #     model.encoder.relu2_1.layers[0].set_weights([f["encoder"][block_name]["layers"][layer_name]["vars"]['0'], f["encoder"][block_name]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'conv2d_1'
            #     model.encoder.relu2_1.layers[2].set_weights([f["encoder"][block_name]["layers"][layer_name]["vars"]['0'], f["encoder"][block_name]["layers"][layer_name]["vars"]['1']])
            #     block_name = 'relu3_1'
            #     layer_name = 'conv2d'
            #     model.encoder.relu3_1.layers[0].set_weights([f["encoder"][block_name]["layers"][layer_name]["vars"]['0'], f["encoder"][block_name]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'conv2d_1'
            #     model.encoder.relu3_1.layers[2].set_weights([f["encoder"][block_name]["layers"][layer_name]["vars"]['0'], f["encoder"][block_name]["layers"][layer_name]["vars"]['1']])
            #     block_name = 'relu4_1'
            #     layer_name = 'conv2d'
            #     model.encoder.relu4_1.layers[0].set_weights([f["encoder"][block_name]["layers"][layer_name]["vars"]['0'], f["encoder"][block_name]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'conv2d_1'
            #     model.encoder.relu4_1.layers[1].set_weights([f["encoder"][block_name]["layers"][layer_name]["vars"]['0'], f["encoder"][block_name]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'conv2d_2'
            #     model.encoder.relu4_1.layers[2].set_weights([f["encoder"][block_name]["layers"][layer_name]["vars"]['0'], f["encoder"][block_name]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'conv2d_3'
            #     model.encoder.relu4_1.layers[4].set_weights([f["encoder"][block_name]["layers"][layer_name]["vars"]['0'], f["encoder"][block_name]["layers"][layer_name]["vars"]['1']])
                
            #     layer_name = 'reflective_conv2d'
            #     model.decoder.decoder.layers[0].set_weights([f["decoder"]["decoder"]["layers"][layer_name]["vars"]['0'], f["decoder"]["decoder"]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'reflective_conv2d_1'
            #     model.decoder.decoder.layers[2].set_weights([f["decoder"]["decoder"]["layers"][layer_name]["vars"]['0'], f["decoder"]["decoder"]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'reflective_conv2d_2'
            #     model.decoder.decoder.layers[3].set_weights([f["decoder"]["decoder"]["layers"][layer_name]["vars"]['0'], f["decoder"]["decoder"]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'reflective_conv2d_3'
            #     model.decoder.decoder.layers[4].set_weights([f["decoder"]["decoder"]["layers"][layer_name]["vars"]['0'], f["decoder"]["decoder"]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'reflective_conv2d_4'
            #     model.decoder.decoder.layers[5].set_weights([f["decoder"]["decoder"]["layers"][layer_name]["vars"]['0'], f["decoder"]["decoder"]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'reflective_conv2d_5'
            #     model.decoder.decoder.layers[7].set_weights([f["decoder"]["decoder"]["layers"][layer_name]["vars"]['0'], f["decoder"]["decoder"]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'reflective_conv2d_6'
            #     model.decoder.decoder.layers[8].set_weights([f["decoder"]["decoder"]["layers"][layer_name]["vars"]['0'], f["decoder"]["decoder"]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'reflective_conv2d_7'
            #     model.decoder.decoder.layers[10].set_weights([f["decoder"]["decoder"]["layers"][layer_name]["vars"]['0'], f["decoder"]["decoder"]["layers"][layer_name]["vars"]['1']])
            #     layer_name = 'reflective_conv2d_8'
            #     model.decoder.decoder.layers[11].set_weights([f["decoder"]["decoder"]["layers"][layer_name]["vars"]['0'], f["decoder"]["decoder"]["layers"][layer_name]["vars"]['1']])




                # for layer in model.layers:
                #     if isinstance(layer, ReflectiveConv2D):
                #         original_conv_name = layer.name.replace("reflective_", "") 
                        
                #         if original_conv_name in layer_names:
                #             weights = f[original_conv_name]
                #             layer.set_weights([weights["kernel:0"][()], weights["bias:0"][()]])
                #         else:
                #             print(f"⚠️ Warning : no weights found for {layer.name}")

            # Remove downloaded files once used to prevent useless disk storage
            shutil.rmtree(tmp_path)

            return model
        except Exception as e :
            shutil.rmtree(tmp_path)
            raise e
        
    def build(self, input_shape) :
        super().build(input_shape)

    def call(self, inputs, training=False):
        content = inputs[0]
        content = keras.applications.vgg19.preprocess_input(content).to(self.device) # Make sure the format of input is compatible with VGG (BGR, mean-centered)
        style = inputs[1]
        style = keras.applications.vgg19.preprocess_input(style).to(self.device)
        resized_style = keras.layers.Resizing(content.shape[1], content.shape[2])(style)

        encoded_content = self.encoder(content) 
        style_full_features = self.encoder(resized_style, all_features=True)
        encoded_style = style_full_features[-1]

        combined_features = self.ada_in(encoded_content, encoded_style)

        decoded_img = self.decoder(combined_features)
        gen_image_full_features = self.encoder(decoded_img, all_features=True)

        self.add_loss(self.loss_.total_loss(gen_image_full_features, style_full_features, combined_features, decoded_img))
        
        return decoded_img
    
    def generate(self, content_img : Path | str | torch.types.Tensor, style_img : Path | str | torch.types.Tensor, show_img = False, show_inputs = False) :
        if isinstance(content_img, (Path, str)) :
            content_img = torch.tensor(keras.utils.img_to_array(keras.utils.load_img(content_img)))
        if isinstance(style_img, (Path, str)) :
            style_img = torch.tensor(keras.utils.img_to_array(keras.utils.load_img(style_img)))
        
        if content_img.max() <= 1.0 :
            content_img *= 255
        if style_img.max() <= 1.0 :
            style_img *= 255
        
        content_img = content_img.type(torch.int)
        style_img = style_img.type(torch.int)

        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)

        IMAGENET_MEANS = torch.tensor([103.939, 116.779, 123.68])
        with torch.no_grad() :
            gen_img = self.call([content_img.unsqueeze(0), style_img.unsqueeze(0)], training=False)[0]
            gen_img += IMAGENET_MEANS.to(self.device) # The decoder generates images normalized around the ImageNet means for each channel
            gen_img = gen_img.flip(dims=[-1]) # Convert BGR to RGB
            gen_img = gen_img.clamp(0,255).cpu().type(torch.int)

        torch.cuda.empty_cache()
        gc.collect()

        if show_img :
            if show_inputs :
                fig, axs = plt.subplots(1,3, figsize=(21,7))
                axs[0].imshow(content_img.cpu())
                axs[0].axis('off')
                axs[1].imshow(style_img.cpu())
                axs[1].axis('off')
                axs[2].imshow(gen_img)
                axs[2].axis('off')
            else :
                plt.imshow(gen_img)
                plt.axis('off')
            plt.show()
        else :
            return gen_img



class AdaINLoss() :
    def __init__(self, style_loss_weight, tv_weight, reduction="sum_over_batch_size") :
        """The custom loss function defined in the AdaIN paper

        Args:
            style_loss_weight (float): Relative weighing of the style loss wrt the content loss
            tv_weight (float): Relative weighing of the total variation loss wrt the content loss
        """
        self.style_loss_weight = style_loss_weight
        self.tv_weight = tv_weight
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
    
    def total_variation_loss(self, img):
        bs_img, h_img, w_img, c_img = img.size()
        tv_h = torch.pow(img[:,1:,:,:]-img[:,:-1,:,:], 2).sum()
        tv_w = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
        return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
    
    def total_loss(self, generated_full_features, style_full_features, adain_output, generated_img) :
        generated_features = generated_full_features[-1]
        if self.tv_weight != 0 : # TV loss is computed only if needed, as it slows the computation
            batched_loss = self.content_loss(generated_features, adain_output) + self.style_loss_weight * self.style_loss(generated_full_features, style_full_features) + self.tv_weight * self.total_variation_loss(generated_img)
        else :
            batched_loss = self.content_loss(generated_features, adain_output) + self.style_loss_weight * self.style_loss(generated_full_features, style_full_features)
        
        match self.reduction :
            case "sum_over_batch_size" | "mean" :
                return batched_loss.mean()
            case "sum" :
                return batched_loss.sum()
            case _ :
                return batched_loss