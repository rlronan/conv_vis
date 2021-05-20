import sys, os
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import math

### Code Reference: Visualizing intermediate activation in Convolutional Neural Networks with Keras
"""
Visualizes activations from layers in CNN.

Activations are computed by passing an image through the model and sampling the output of each layer.

Usage:     plot_metrics = visualize_activations(num_layers=6,sample_image=input_image,mode='test',layers=[1,2])
           loss, acc = model.evaluate(testX, testY, verbose=2, callbacks = [plot_metrics])

Params: Model
Returns: none

"""
class visualize_activations(tf.keras.callbacks.Callback):
    def __init__(self,sample_image=None,num_layers=None,mode=None,interval=None,layers=[]):
      self.num_layers=num_layers
      self.sample_image=sample_image
      self.mode=mode
      self.interval=interval
      self.layers = layers
   


    def on_epoch_end(self, epoch, logs={}):
        if self.mode=='epoch' and (epoch%self.interval == 0):
            self.vis_layers()

    def on_batch_end(self, batch, logs={}):
        if self.mode=='batch' and (batch%self.interval == 0):
            self.vis_layers()

    def on_test_end(self,logs={}):
        if self.mode=='test':
            self.vis_layers()


    def vis_layers(self):

        model = self.model
        num_layers = self.num_layers

        if self.sample_image is None:
            randomNoise = np.random.rand(1,input_width,input_witdth,3)
            randomNoise = randomNoise * 255
        else:
            randomNoise = self.sample_image
            if len(randomNoise.shape) == 3:
                randomNoise = tf.expand_dims(randomNoise, axis=0)


        activation_dims = model.layers[0].output
        input_width = activation_dims[1]

        if len(self.layers) == 0:
            layer_names = []
            for layer in model.layers[:num_layers]:
                layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
            layer_outputs = [layer.output for layer in model.layers[1:num_layers+1]] # Gathers the outputs of the layers we want 

        else:
            layer_names = []
            layer_outputs = []
            for layer_num in self.layers:
                layer_names.append(model.layers[layer_num].name)
                layer_outputs.append(model.layers[layer_num].output)


        activation_model = Model(inputs=model.input, outputs=layer_outputs) # Isolates the model layers from our model
        activations = activation_model.predict(randomNoise) # Returns a list of five Numpy arrays: one array per layer activation

        images_per_row = 16


        for layer_name, layer_activation in zip(layer_names, activations): # Iterates over every layer
            n_features = layer_activation.shape[-1] # Number of features in the feature map
            output_size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
            n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
            layer_vis = np.zeros((output_size * n_cols, images_per_row * output_size))


            for col in range(n_cols):
                for row in range(images_per_row):
                    feature = layer_activation[0, :, :, col * images_per_row + row]
                    # Scale and transform the activation for display
                    feature -= feature.mean() # Subtract the mean
                    feature /= feature.std() # Normalize

                    # Don't allow the intensity values to be too large (max 200... over 200 is harsh to look at)
                    feature *= 50
                    feature += 150
                    feature = np.clip(feature, 0, 255).astype('uint8')
                    # displays a panel of
                    layer_vis[col * output_size : (col + 1) * output_size,
                                row * output_size : (row + 1) * output_size] = feature
            scale = 1. / output_size
            plt.figure(figsize=(scale * layer_vis.shape[1],
                                scale * layer_vis.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(layer_vis, aspect='auto', cmap='plasma')
            plt.show()

