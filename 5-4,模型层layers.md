# 5-4,模型层layers

深度学习模型一般由各种模型层组合而成。

tf.keras.layers内置了非常丰富的各种功能的模型层。例如，

layers.Dense,layers.Flatten,layers.Input,layers.DenseFeature,layers.Dropout

layers.Conv2D,layers.MaxPooling2D,layers.Conv1D

layers.Embedding,layers.GRU,layers.LSTM,layers.Bidirectional等等。

如果这些内置模型层不能够满足需求，我们也可以通过编写tf.keras.Lambda匿名模型层或继承tf.keras.layers.Layer基类构建自定义的模型层。

其中tf.keras.Lambda匿名模型层只适用于构造没有学习参数的模型层。

```python

```

### 一，内置模型层


一些常用的内置模型层简单介绍如下。

**基础层**

* Dense：密集连接层。参数个数 = 输入层特征数× 输出层特征数(weight)＋ 输出层特征数(bias)

* Activation：激活函数层。一般放在Dense层后面，等价于在Dense层中指定activation。

* Dropout：随机置零层。训练期间以一定几率将输入置0，一种正则化手段。

* BatchNormalization：批标准化层。通过线性变换将输入批次缩放平移到稳定的均值和标准差。可以增强模型对输入不同分布的适应性，加快模型训练速度，有轻微正则化效果。一般在激活函数之前使用。

* SpatialDropout2D：空间随机置零层。训练期间以一定几率将整个特征图置0，一种正则化手段，有利于避免特征图之间过高的相关性。

* Input：输入层。通常使用Functional API方式构建模型时作为第一层。

* DenseFeature：特征列接入层，用于接收一个特征列列表并产生一个密集连接层。

* Flatten：压平层，用于将多维张量压成一维。

* Reshape：形状重塑层，改变输入张量的形状。

* Concatenate：拼接层，将多个张量在某个维度上拼接。

* Add：加法层。

* Subtract： 减法层。

* Maximum：取最大值层。

* Minimum：取最小值层。


**卷积网络相关层**

* Conv1D：普通一维卷积，常用于文本。参数个数 = 输入通道数×卷积核尺寸(如3)×卷积核个数

* Conv2D：普通二维卷积，常用于图像。参数个数 = 输入通道数×卷积核尺寸(如3乘3)×卷积核个数

* Conv3D：普通三维卷积，常用于视频。参数个数 = 输入通道数×卷积核尺寸(如3乘3乘3)×卷积核个数

* SeparableConv2D：二维深度可分离卷积层。不同于普通卷积同时对区域和通道操作，深度可分离卷积先操作区域，再操作通道。即先对每个通道做独立卷积操作区域，再用1乘1卷积跨通道组合操作通道。参数个数 = 输入通道数×卷积核尺寸 + 输入通道数×1×1×输出通道数。深度可分离卷积的参数数量一般远小于普通卷积，效果一般也更好。

* DepthwiseConv2D：二维深度卷积层。仅有SeparableConv2D前半部分操作，即只操作区域，不操作通道，一般输出通道数和输入通道数相同，但也可以通过设置depth_multiplier让输出通道为输入通道的若干倍数。输出通道数 = 输入通道数 × depth_multiplier。参数个数 = 输入通道数×卷积核尺寸× depth_multiplier。

* Conv2DTranspose：二维卷积转置层，俗称反卷积层。并非卷积的逆操作，但在卷积核相同的情况下，当其输入尺寸是卷积操作输出尺寸的情况下，卷积转置的输出尺寸恰好是卷积操作的输入尺寸。

* LocallyConnected2D: 二维局部连接层。类似Conv2D，唯一的差别是没有空间上的权值共享，所以其参数个数远高于二维卷积。

* MaxPool2D: 二维最大池化层。也称作下采样层。池化层无可训练参数，主要作用是降维。

* AveragePooling2D: 二维平均池化层。

* GlobalMaxPool2D: 全局最大池化层。每个通道仅保留一个值。一般从卷积层过渡到全连接层时使用，是Flatten的替代方案。

* GlobalAvgPool2D: 全局平均池化层。每个通道仅保留一个值。


**循环网络相关层**

* Embedding：嵌入层。一种比Onehot更加有效的对离散特征进行编码的方法。一般用于将输入中的单词映射为稠密向量。嵌入层的参数需要学习。

* LSTM：长短记忆循环网络层。最普遍使用的循环网络层。具有携带轨道，遗忘门，更新门，输出门。可以较为有效地缓解梯度消失问题，从而能够适用长期依赖问题。设置return_sequences = True时可以返回各个中间步骤输出，否则只返回最终输出。

* GRU：门控循环网络层。LSTM的低配版，不具有携带轨道，参数数量少于LSTM，训练速度更快。

* SimpleRNN：简单循环网络层。容易存在梯度消失，不能够适用长期依赖问题。一般较少使用。

* ConvLSTM2D：卷积长短记忆循环网络层。结构上类似LSTM，但对输入的转换操作和对状态的转换操作都是卷积运算。

* Bidirectional：双向循环网络包装器。可以将LSTM，GRU等层包装成双向循环网络。从而增强特征提取能力。

* RNN：RNN基本层。接受一个循环网络单元或一个循环单元列表，通过调用tf.keras.backend.rnn函数在序列上进行迭代从而转换成循环网络层。

* LSTMCell：LSTM单元。和LSTM在整个序列上迭代相比，它仅在序列上迭代一步。可以简单理解LSTM即RNN基本层包裹LSTMCell。

* GRUCell：GRU单元。和GRU在整个序列上迭代相比，它仅在序列上迭代一步。

* SimpleRNNCell：SimpleRNN单元。和SimpleRNN在整个序列上迭代相比，它仅在序列上迭代一步。

* AbstractRNNCell：抽象RNN单元。通过对它的子类化用户可以自定义RNN单元，再通过RNN基本层的包裹实现用户自定义循环网络层。

* Attention：Dot-product类型注意力机制层。可以用于构建注意力模型。

* AdditiveAttention：Additive类型注意力机制层。可以用于构建注意力模型。

* TimeDistributed：时间分布包装器。包装后可以将Dense、Conv2D等作用到每一个时间片段上。

```python

```

### 二，自定义模型层


如果自定义模型层没有需要被训练的参数，一般推荐使用Lamda层实现。

如果自定义模型层有需要被训练的参数，则可以通过对Layer基类子类化实现。

Lambda层由于没有需要被训练的参数，只需要定义正向传播逻辑即可，使用比Layer基类子类化更加简单。

Lambda层的正向逻辑可以使用Python的lambda函数来表达，也可以用def关键字定义函数来表达。

```python
import tensorflow as tf
from tensorflow.keras import layers,models,regularizers

mypower = layers.Lambda(lambda x:tf.math.pow(x,2))
mypower(tf.range(5))
```

```
<tf.Tensor: shape=(5,), dtype=int32, numpy=array([ 0,  1,  4,  9, 16], dtype=int32)>
```


Layer的子类化一般需要重新实现初始化方法，Build方法和Call方法。下面是一个简化的线性层的范例，类似Dense.

```python
class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
    
    #build方法一般定义Layer需要被训练的参数。    
    def build(self, input_shape): 
        self.w = self.add_weight("w",shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True) #注意必须要有参数名称"w",否则会报错
        self.b = self.add_weight("b",shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(Linear,self).build(input_shape) # 相当于设置self.built = True

    #call方法一般定义正向传播运算逻辑，__call__方法调用了它。  
    @tf.function
    def call(self, inputs): 
        return tf.matmul(inputs, self.w) + self.b
    
    #如果要让自定义的Layer通过Functional API 组合成模型时可以被保存成h5模型，需要自定义get_config方法。
    def get_config(self):  
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

```

```python
linear = Linear(units = 8)
print(linear.built)
#指定input_shape，显式调用build方法，第0维代表样本数量，用None填充
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
#如果built = False，调用__call__时会先调用build方法, 再调用call方法。
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
#注意该处的input_shape会被模型加工，无需使用None代表样本数量维
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


# 保存成 h5模型
model.save("./data/linear_model.h5",save_format = "h5")
model_loaded_keras = tf.keras.models.load_model(
    "./data/linear_model.h5",custom_objects={"Linear":Linear})
print(model_loaded_keras.predict(tf.constant([[3.0,2.0],[4.0,5.0]])))


# 保存成 tf模型
model.save("./data/linear_model",save_format = "tf")
model_loaded_tf = tf.keras.models.load_model("./data/linear_model")
print(model_loaded_tf.predict(tf.constant([[3.0,2.0],[4.0,5.0]])))

```

```
[[-0.04092304]
 [-0.06150477]]
[[-0.04092304]
 [-0.06150477]]
INFO:tensorflow:Assets written to: ./data/linear_model/assets
[[-0.04092304]
 [-0.06150477]]
```


如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"算法美食屋"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![算法美食屋二维码.jpg](./data/算法美食屋二维码.jpg)
