# EasyVIZ
![Header Image](/sample_images/header_image.png)

Easy to use visualizations for convolutional networks, and metrics for those visualizations.

For a demo of the package please see ![EasyVIZ Demo](https://github.com/rlronan/cnn_vis_demo).

The repository currently contains code to generate and save maximum mean filter activations for convolutional neural networks, 
and code to compute the 2D-Entropy (also known as delentropy) of those filters.

## How To Use

feature_vis.py contains a standalone function `log_conv_features` for Tensorflow models, and a Tensorflow Callback function `log_conv_features_callback`. 
These functions, more so when `entropy=True` is set, are quite computationally intensive. 
Generating a single maximum mean filter activations typically involves hundreds of an iterations of an image through (part) of a model.
These functions run much faster when no `preprocesssing_func` (see below) is required, or the `default` or `255` values are used for the preprocessing function. 

Both functions require a list of convolutional layer numbers (indices) to visualize.
These indices are NOT the same as the layer index in the model. 
The first convolutional layer always convolutional layer number of zero, and the second always has a convolutional number of 1, and so on.

You can display these numbers for your model by using the function `show_conv_layers(model=my_model)` from utils.py.

If your model expect inputs in the [0,1] range, you can leave `preprocess_func=None`. Otherwise call either logging function with:

  `preprocess_func='default'` for images in the [-1,1] range.
  
  `preprocess_func='255'` for images in the [0,255] range (although they must be floats).
  
  `preprocess_func=tf.keras.applications.[model_name].preprocess_input` for use on a pretrained tf.keras.applications model. You can define your own preprocessing function as long as it accepts images in the [0,255] range, and functions analagously to ![tf.keras.applications.mobilenet.preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/preprocess_input)
  
### Example Standalone Usage
```
>>> model = tf.keras.applications.VGG16()
>>> show_conv_layers(model=my_model)
conv layer #, 	 layer name, 	 layer index in model
0 		 block1_conv1 		 1
1 		 block1_conv2 		 2
2 		 block2_conv1 		 4
.
.
.
10 		 block5_conv1 		 15
11 		 block5_conv2 		 16
12 		 block5_conv3 		 17
>>> log_conv_features(model, 
                      layer_nums=[0,1,6,5,11,12], 
                      preprocess_func=tf.keras.applications.vgg16.preprocess_input,
                      directory=pathlib.Path('./feature_logs/'), 
                      filter_indices=np.arange(16), # default value
                      iterations=250,               # default value
                      step_size=1,                  # default value
                      resizes=10,                   # default value
                      resize_factor=1.2,            # default value
                      entropy=True                  # default value
                      )                 
```

### Example Callback Usage
```
>>> model = tf.keras.applications.VGG16()
>>> show_conv_layers(model=model)
conv layer #, 	 layer name, 	 layer index in model
0 		 block1_conv1 		 1
1 		 block1_conv2 		 2
2 		 block2_conv1 		 4
.
.
.
10 		 block5_conv1 		 15
11 		 block5_conv2 		 16
12 		 block5_conv3 		 17
>>> feature_callback = log_conv_features_callback(
        log_dir=pathlib.Path('./feature_logs/'),
        update_freq='epoch',                        # default value
        update_freq_val=5,
        layer_nums=[0,1,2,10,11,12],
        preprocess_func=tf.keras.applications.vgg16.preprocess_input,
        clip=True,                                  # default value
        entropy=True                                # default value
        )
>>> history = model.fit(train_dataset, epochs=20, validation_data=val_dataset,
                        callbacks=[feature_callback])
```


## Examples
#### Maximum Mean Filter Activations With VGG16
The maximum mean filter activations from the first sixteen filters of block3_conv3, and block4_conv1 in a VGG16 network pretrained on the imagenet dataset.
<p float="left">
  <img src="/sample_images/block3_conv3_filter_6.75103142.png" width="410" />
  <img src="/sample_images/block4_conv1_filter_6.645063669.png" width="410" /> 
</p>




#### 2D-Entropy Examples With VGG16
Filters from Block 3, Conv 1. While they both appear to be detecting edges in the direction from top left to bottom right, the second filter also shows detections in the orthogonal direction. 
<p float="left">
  <img src="/sample_images/block3_conv1_4_entropy7.067398583164607.png" width="410" />
  <img src="/sample_images/block3_conv1_5_entropy7.373630438611548.png" width="410" /> 
</p>

## Requirements:

tensorflow 2.x

tensorflow_addons compatible with the installed tensorflow version

scipy

pathlib

imageio

matplotlib

numpy

