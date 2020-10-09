# 5-4 layers

The deep learning model usually consists of various layers.

tf.keras.layers contains a large amount of models with various functions, such as:

`layers.Dense`, `layers.Flatten`, `layers.Input`, `layers.DenseFeature`, `layers.Dropout`

`layers.Conv2D`, `layers.MaxPooling2D`, `layers.Conv1D`

`layers.Embedding`, `layers.GRU`, `layers.LSTM`, `layers.Bidirectional`, etc.

In case these pre-defined layers are insufficient for modeling, the users are able to write anonymous layer `tf.keras.Lambda` or write customized layer through inheriting `tf.keras.layers.Layer`.

Note that `tf.keras.Lambda` is only for the layers without any trainable parameter.

```python

```

### 1. Pre-defined Layer


Here are the introductions for the most popular layers.

**Fundamental layers**

* Dense: Densely connected layer. # of parameters = # of features of the input layer × # of weight + # of bias.

* Activation: Activation function layer. Usually placed after the Dense layer for specify the activation function in the Dense layer.

* Dropout: Dropout layer. Setting the inputs as zero randomly during the training stage, which is a method of regularization.

* BatchNormalization:Layer for batch normalization. It scale and translate the batches into stable mean and standad deviation through linear transformation. This lead to the model adaptive to the various distribution of the input, which is mild regularization that accelerates training. This layer is usually applied before the activation function.

* SpatialDropout2D:Spatial dropout layer. Setting the whole feature map into zero with certain possibilities during training, which is a regularization to avoid high correlation between feature maps.

* Input:Input layer. Usually it is the first layer when modelling through functional API.

* DenseFeature:Layer that provides connection to the feature columns, which is used to accept a list of feature columns and generate a densely connected layer.

* Flatten:Flatten layer, used for flattening multi-dimensional tensor into one-dimension.

* Reshape:Reshape layer, reform the shape of the input tensor.

* Concatenate:Concatenating layer to link multiple tensors along the specific dimension.

* Add: Adding layer.

* Subtract: Subtracting layer.

* Maximum: Layer for maximum value.

* Minimum: Layer for minimum value.


**Layers for the convoelutional network.**

* `Conv1D`: Layer of 1D convolution, ususlly for text. # of parameters = # of input channels × # of kernel size (e.g. 3) × # of kernels.

* `Conv2D`: Layer of 2D convolution, ususlly for image. # of parameters = # of input channels × # of kernel size (e.g. 3×3) × # of kernels.

* `Conv3D`: Layer of 3D convolution, ususlly for video. # of parameters = # of input channels × # of kernel size (e.g. 3×3×3) × # of kernels.

* `SeparableConv2D`: Depthwise 2D separable covolution. Unlike the traditional convolution, separable convolutions consist in first performing a depthwise spatial convolution (which acts on each input channel separately) followed by a pointwise convolution which mixes together the resulting output channels. # of parameters = # of input channels × size of convolutional kernel + # of input channels × 1 × 1 × # of output channels. Usually, depthwise separable convolution has a much smaller number of parameters with a better performance.

* `DepthwiseConv2D`: Depthwise 2D convolution. Depthwise convolutions consists in performing just the first step in a depthwise separable convolution (which acts on each input channel separately). Usually the # of input and output channels are the same, but can be altered through the `depth_multiplier` argument to control how many output channels are generated per input channel in the depthwise step. # of output channles =  # of input channles × depth_multiplier. # of parameters = # of input channels × size of kernel × `depth_multiplier`.

* `Conv2DTranspose`:2D Transposed convolution layer (sometimes called Deconvolution), but this is not the inverse operation of convolution. When the kernal maintains the same as convolution, and given the input size the same as the output size of convolution, then the output size of the transposed convolution is the same as the input size of convolution.

* `LocallyConnected2D`: Locally-connected layer for 2D inputs. This layer works similarly to the `Conv2D` layer, except that weights are unshared, thus has much more parameters than `Conv2D`.

* `MaxPooling2D`: 2D max pooling layer, also called down-sampling layer. This layer is for reducing dimension without any trainable prameter.

* `AveragePooling2D`: 2D average pooling layer.

* `GlobalMaxPool2D`: Global max pooling layer. Only one value preserved for each channel, usually used between convolution layer and fully connected layer, which is a substitution of `Flatten`.

* `GlobalAvgPool2D`: Global average pooling layer. Only one value preserved for each channel.


**Recursive network related layers**

* `Embedding`: Embedding layer, provides an encoding with higher efficiency than one-hot for discrete features. It is usually used for projecting input words into dense vectors. Training is required for the parameters in the embedding layer.

* `LSTM`: Long Short-Term Memory layer, which is the most popular layer for the recursive network. It contains carry track and is composed of a cell, an input gate, an output gate and a forget gate, which significantly alleviate the problem of gradient vanishing and thus applicable for the problem of long-term dependency. All the middle results could be observed when the patameter `return_sequences` is set as `True`; in the opposite case only the final result is returned.

* `GRU`: Gated Recursive Unit Layer, a simplified version of LSTM without carry track, thus has less parameters and could be trained faster.

* `SimpleRNN`: Simple Recursive Neural Network layer. It is not popular due to the problem of gradient vanishing, which made it inapplicable to the long-dependence.

* `ConvLSTM2D`: Convolutional LSTM layer. Has similar structure to LSTM but the conversion operation to both input and status are convolution.

* `Bidirectional`: Bi-directional wrapper for RNNs, for wrapping layers (such as LSTM and GRU) into bi-directional RNN, enhancing the capability of feature extraction.

* `RNN`: Base class of RNN. It accepts an RNN cell or a list of RNN cells, and iterate on the sequence to convert the input as an RNN through the calling of `tf.keras.backend.rnn` function.

* `LSTMCell`: LSTM cell. Unlike iterating across the whole sequence as `LSTM`, it only iterate once on the sequence. It would be more intuitive to understand the LSTM equals to `LSTMCell` wrapped by the base layer `RNN`.

* `GRUCell`: GRU cell. Unlike iterating across the whole sequence as `GRU`, it only iterate once on the sequence.

* `SimpleRNNCell`: SimpleRNN cell. Unlike iterating across the whole sequence as `SimpleRNN`, it only iterate once on the sequence.

* `AbstractRNNCell`: Abstract RNN Cell. It allows user to customize RNN cell through inheritance and subsequently construct the RNN layer through wrapping these RNN cells by RNN base layer.

* `Attention`: Dot-product attention layer, a.k.a. Luong-style attention, for constructing attention model.

* `AdditiveAttention`: Additive attention layer, a.k.a. Bahdanau-style attention, for constructing attention model.

* `TimeDistributed`: Time distributed wrapper, allows applying `Dense`, `Conv2D` to each time segment.

```python

```

### 2. Customized Model Layer


It is recommended to use `Lambda` layer for customized model layer without trainable parameter.

For the customized model layer with trainable parameters, it is recommended to inherite from the base class `Layer`.

We only need to define forward propagation for `Lambda` layer since there is no trainable parameter, thus it is easier in application comparing to the inheritance from base class `Layer`.

The forward propagation of `Lambda` layer could be expressed using `lambda` function or keyword `def` in Python.

```python
import tensorflow as tf
from tensorflow.keras import layers,models,regularizers

mypower = layers.Lambda(lambda x:tf.math.pow(x,2))
mypower(tf.range(5))
```

```
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([ 0,  1,  4,  9, 16], dtype=int32)>
```


Inheriting from `Layer` needs re-implementation of the initialization, `build`and `call` methods. Here is an example of simplified linear layer, which is similar to `Dense`.

```python
class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
    
    # The trainable parameters are defined in build method
    def build(self, input_shape): 
        self.w = self.add_weight("w",shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True) # Parameter named "w" is compulsory or an error will be thrown out
        self.b = self.add_weight("b",shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(Linear,self).build(input_shape) # Identical to self.built = True

    # The logic of forward propagation is defined in call method, and is called by __call__ method
    @tf.function
    def call(self, inputs): 
        return tf.matmul(inputs, self.w) + self.b
    
    # Use customized get-config method to save the model as h5 format, specifically for the model composed through Functional API with customized Layer
    def get_config(self):  
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

```

```python
linear = Linear(units = 8)
print(linear.built)
# Specify input_shape, call build method; the 0th dimension is for the number of samples, should be filled with None.
linear.build(input_shape = (None,16)) 
print(linear.built)
```

```
False
True
```

```python
linear = Linear(units = 8)
print(linear.built)
linear.build(input_shape = (None,16)) 
print(linear.compute_output_shape(input_shape = (None,16)))
```

```
False
(None, 8)
```

```python
linear = Linear(units = 16)
print(linear.built)
# If built = False, the __call__ method will call "build" method first, then call "call" method
linear(tf.random.uniform((100,64))) 
print(linear.built)
config = linear.get_config()
print(config)
```

```
False
True
{'name': 'linear_3', 'trainable': True, 'dtype': 'float32', 'units': 16}
```

```python
tf.keras.backend.clear_session()

model = models.Sequential()
# Note: the input_shape here will be modified by the model, so we don't have to fill None in the dimension representing the number of samples.
model.add(Linear(units = 1,input_shape = (2,)))  
print("model.input_shape: ",model.input_shape)
print("model.output_shape: ",model.output_shape)
model.summary()
```

```
model.input_shape:  (None, 2)
model.output_shape:  (None, 1)
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
linear (Linear)              (None, 1)                 3         
=================================================================
Total params: 3
Trainable params: 3
Non-trainable params: 0
_________________________________________________________________
```

```python
model.compile(optimizer = "sgd",loss = "mse",metrics=["mae"])
print(model.predict(tf.constant([[3.0,2.0],[4.0,5.0]])))


# Save as h5 formatted model
model.save("../data/linear_model.h5",save_format = "h5")
model_loaded_keras = tf.keras.models.load_model(
    "../data/linear_model.h5",custom_objects={"Linear":Linear})
print(model_loaded_keras.predict(tf.constant([[3.0,2.0],[4.0,5.0]])))


# Save as tf formatted model
model.save("../data/linear_model",save_format = "tf")
model_loaded_tf = tf.keras.models.load_model("../data/linear_model")
print(model_loaded_tf.predict(tf.constant([[3.0,2.0],[4.0,5.0]])))

```

```
[[-0.04092304]
 [-0.06150477]]
[[-0.04092304]
 [-0.06150477]]
INFO:tensorflow:Assets written to: ../data/linear_model/assets
[[-0.04092304]
 [-0.06150477]]
```


Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)
