# 6-5,使用TPU训练模型

如果想尝试使用Google Colab上的TPU来训练模型，也是非常方便，仅需添加6行代码。

在Colab笔记本中：修改->笔记本设置->硬件加速器 中选择 TPU

注：以下代码只能在Colab 上才能正确执行。

可通过以下colab链接测试效果《tf_TPU》：

https://colab.research.google.com/drive/1XCIhATyE1R7lq6uwFlYlRsUr5d9_-r1s


```python
%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import * 
```

### 一，准备数据

```python
MAX_LEN = 300
BATCH_SIZE = 32
(x_train,y_train),(x_test,y_test) = datasets.reuters.load_data()
x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=MAX_LEN)
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=MAX_LEN)

MAX_WORDS = x_train.max()+1
CAT_NUM = y_train.max()+1

ds_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)) \
          .shuffle(buffer_size = 1000).batch(BATCH_SIZE) \
          .prefetch(tf.data.experimental.AUTOTUNE).cache()
   
ds_test = tf.data.Dataset.from_tensor_slices((x_test,y_test)) \
          .shuffle(buffer_size = 1000).batch(BATCH_SIZE) \
          .prefetch(tf.data.experimental.AUTOTUNE).cache()
```

### 二，定义模型

```python
tf.keras.backend.clear_session()
def create_model():
    
    model = models.Sequential()

    model.add(layers.Embedding(MAX_WORDS,7,input_length=MAX_LEN))
    model.add(layers.Conv1D(filters = 64,kernel_size = 5,activation = "relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(filters = 32,kernel_size = 3,activation = "relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(CAT_NUM,activation = "softmax"))
    return(model)

def compile_model(model):
    model.compile(optimizer=optimizers.Nadam(),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[metrics.SparseCategoricalAccuracy(),metrics.SparseTopKCategoricalAccuracy(5)]) 
    return(model)
```

```python

```

### 三，训练模型

```python
#增加以下6行代码
import os
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
    model = create_model()
    model.summary()
    model = compile_model(model)
    
```

```
WARNING:tensorflow:TPU system 10.26.134.242:8470 has already been initialized. Reinitializing the TPU can cause previously created variables on TPU to be lost.
WARNING:tensorflow:TPU system 10.26.134.242:8470 has already been initialized. Reinitializing the TPU can cause previously created variables on TPU to be lost.
INFO:tensorflow:Initializing the TPU system: 10.26.134.242:8470
INFO:tensorflow:Initializing the TPU system: 10.26.134.242:8470
INFO:tensorflow:Clearing out eager caches
INFO:tensorflow:Clearing out eager caches
INFO:tensorflow:Finished initializing TPU system.
INFO:tensorflow:Finished initializing TPU system.
INFO:tensorflow:Found TPU system:
INFO:tensorflow:Found TPU system:
INFO:tensorflow:*** Num TPU Cores: 8
INFO:tensorflow:*** Num TPU Cores: 8
INFO:tensorflow:*** Num TPU Workers: 1
INFO:tensorflow:*** Num TPU Workers: 1
INFO:tensorflow:*** Num TPU Cores Per Worker: 8
INFO:tensorflow:*** Num TPU Cores Per Worker: 8
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:localhost/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
INFO:tensorflow:*** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 0, 0)
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 300, 7)            216874    
_________________________________________________________________
conv1d (Conv1D)              (None, 296, 64)           2304      
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 148, 64)           0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 146, 32)           6176      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 73, 32)            0         
_________________________________________________________________
flatten (Flatten)            (None, 2336)              0         
_________________________________________________________________
dense (Dense)                (None, 46)                107502    
=================================================================
Total params: 332,856
Trainable params: 332,856
Non-trainable params: 0
_________________________________________________________________
```

```python
history = model.fit(ds_train,validation_data = ds_test,epochs = 10)
```

```
Train for 281 steps, validate for 71 steps
Epoch 1/10
281/281 [==============================] - 12s 43ms/step - loss: 3.4466 - sparse_categorical_accuracy: 0.4332 - sparse_top_k_categorical_accuracy: 0.7180 - val_loss: 3.3179 - val_sparse_categorical_accuracy: 0.5352 - val_sparse_top_k_categorical_accuracy: 0.7195
Epoch 2/10
281/281 [==============================] - 6s 20ms/step - loss: 3.3251 - sparse_categorical_accuracy: 0.5405 - sparse_top_k_categorical_accuracy: 0.7302 - val_loss: 3.3082 - val_sparse_categorical_accuracy: 0.5463 - val_sparse_top_k_categorical_accuracy: 0.7235
Epoch 3/10
281/281 [==============================] - 6s 20ms/step - loss: 3.2961 - sparse_categorical_accuracy: 0.5729 - sparse_top_k_categorical_accuracy: 0.7280 - val_loss: 3.3026 - val_sparse_categorical_accuracy: 0.5499 - val_sparse_top_k_categorical_accuracy: 0.7217
Epoch 4/10
281/281 [==============================] - 5s 19ms/step - loss: 3.2751 - sparse_categorical_accuracy: 0.5924 - sparse_top_k_categorical_accuracy: 0.7276 - val_loss: 3.2957 - val_sparse_categorical_accuracy: 0.5543 - val_sparse_top_k_categorical_accuracy: 0.7217
Epoch 5/10
281/281 [==============================] - 5s 19ms/step - loss: 3.2655 - sparse_categorical_accuracy: 0.6008 - sparse_top_k_categorical_accuracy: 0.7290 - val_loss: 3.3022 - val_sparse_categorical_accuracy: 0.5490 - val_sparse_top_k_categorical_accuracy: 0.7231
Epoch 6/10
281/281 [==============================] - 5s 19ms/step - loss: 3.2616 - sparse_categorical_accuracy: 0.6041 - sparse_top_k_categorical_accuracy: 0.7295 - val_loss: 3.3015 - val_sparse_categorical_accuracy: 0.5503 - val_sparse_top_k_categorical_accuracy: 0.7235
Epoch 7/10
281/281 [==============================] - 6s 21ms/step - loss: 3.2595 - sparse_categorical_accuracy: 0.6059 - sparse_top_k_categorical_accuracy: 0.7322 - val_loss: 3.3064 - val_sparse_categorical_accuracy: 0.5454 - val_sparse_top_k_categorical_accuracy: 0.7266
Epoch 8/10
281/281 [==============================] - 6s 21ms/step - loss: 3.2591 - sparse_categorical_accuracy: 0.6063 - sparse_top_k_categorical_accuracy: 0.7327 - val_loss: 3.3025 - val_sparse_categorical_accuracy: 0.5481 - val_sparse_top_k_categorical_accuracy: 0.7231
Epoch 9/10
281/281 [==============================] - 5s 19ms/step - loss: 3.2588 - sparse_categorical_accuracy: 0.6062 - sparse_top_k_categorical_accuracy: 0.7332 - val_loss: 3.2992 - val_sparse_categorical_accuracy: 0.5521 - val_sparse_top_k_categorical_accuracy: 0.7257
Epoch 10/10
281/281 [==============================] - 5s 18ms/step - loss: 3.2577 - sparse_categorical_accuracy: 0.6073 - sparse_top_k_categorical_accuracy: 0.7363 - val_loss: 3.2981 - val_sparse_categorical_accuracy: 0.5516 - val_sparse_top_k_categorical_accuracy: 0.7306
CPU times: user 18.9 s, sys: 3.86 s, total: 22.7 s
Wall time: 1min 1s
```

```python

```

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"算法美食屋"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![算法美食屋二维码.jpg](./data/算法美食屋二维码.jpg)

```python

```
