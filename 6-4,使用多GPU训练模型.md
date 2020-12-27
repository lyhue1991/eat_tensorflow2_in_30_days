# 6-4,使用多GPU训练模型

如果使用多GPU训练模型，推荐使用内置fit方法，较为方便，仅需添加2行代码。

在Colab笔记本中：修改->笔记本设置->硬件加速器 中选择 GPU

注：以下代码只能在Colab 上才能正确执行。

可通过以下colab链接测试效果《tf_多GPU》：

https://colab.research.google.com/drive/1j2kp_t0S_cofExSN7IyJ4QtMscbVlXU-



MirroredStrategy过程简介：

* 训练开始前，该策略在所有 N 个计算设备上均各复制一份完整的模型；
* 每次训练传入一个批次的数据时，将数据分成 N 份，分别传入 N 个计算设备（即数据并行）；
* N 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据的梯度；
* 使用分布式计算的 All-reduce 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都有了所有设备的梯度之和；
* 使用梯度求和的结果更新本地变量（镜像变量）；
* 当所有设备均更新本地变量后，进行下一轮训练（即该并行策略是同步的）。

```python
%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import * 
```

```python
#此处在colab上使用1个GPU模拟出两个逻辑GPU进行多GPU训练
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 设置两个逻辑GPU模拟多GPU训练
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
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

### 三，训练模型

```python
#增加以下两行代码
strategy = tf.distribute.MirroredStrategy()  
with strategy.scope(): 
    model = create_model()
    model.summary()
    model = compile_model(model)
    
history = model.fit(ds_train,validation_data = ds_test,epochs = 10)  
```

```
WARNING:tensorflow:NCCL is not supported when using virtual GPUs, fallingback to reduction to one device
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1')
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
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
Train for 281 steps, validate for 71 steps
Epoch 1/10
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
281/281 [==============================] - 15s 53ms/step - loss: 2.0270 - sparse_categorical_accuracy: 0.4653 - sparse_top_k_categorical_accuracy: 0.7481 - val_loss: 1.7517 - val_sparse_categorical_accuracy: 0.5481 - val_sparse_top_k_categorical_accuracy: 0.7578
Epoch 2/10
281/281 [==============================] - 4s 14ms/step - loss: 1.5206 - sparse_categorical_accuracy: 0.6045 - sparse_top_k_categorical_accuracy: 0.7938 - val_loss: 1.5715 - val_sparse_categorical_accuracy: 0.5993 - val_sparse_top_k_categorical_accuracy: 0.7983
Epoch 3/10
281/281 [==============================] - 4s 14ms/step - loss: 1.2178 - sparse_categorical_accuracy: 0.6843 - sparse_top_k_categorical_accuracy: 0.8547 - val_loss: 1.5232 - val_sparse_categorical_accuracy: 0.6327 - val_sparse_top_k_categorical_accuracy: 0.8112
Epoch 4/10
281/281 [==============================] - 4s 13ms/step - loss: 0.9127 - sparse_categorical_accuracy: 0.7648 - sparse_top_k_categorical_accuracy: 0.9113 - val_loss: 1.6527 - val_sparse_categorical_accuracy: 0.6296 - val_sparse_top_k_categorical_accuracy: 0.8201
Epoch 5/10
281/281 [==============================] - 4s 14ms/step - loss: 0.6606 - sparse_categorical_accuracy: 0.8321 - sparse_top_k_categorical_accuracy: 0.9525 - val_loss: 1.8791 - val_sparse_categorical_accuracy: 0.6158 - val_sparse_top_k_categorical_accuracy: 0.8219
Epoch 6/10
281/281 [==============================] - 4s 14ms/step - loss: 0.4919 - sparse_categorical_accuracy: 0.8799 - sparse_top_k_categorical_accuracy: 0.9725 - val_loss: 2.1282 - val_sparse_categorical_accuracy: 0.6037 - val_sparse_top_k_categorical_accuracy: 0.8112
Epoch 7/10
281/281 [==============================] - 4s 14ms/step - loss: 0.3947 - sparse_categorical_accuracy: 0.9051 - sparse_top_k_categorical_accuracy: 0.9814 - val_loss: 2.3033 - val_sparse_categorical_accuracy: 0.6046 - val_sparse_top_k_categorical_accuracy: 0.8094
Epoch 8/10
281/281 [==============================] - 4s 14ms/step - loss: 0.3335 - sparse_categorical_accuracy: 0.9207 - sparse_top_k_categorical_accuracy: 0.9863 - val_loss: 2.4255 - val_sparse_categorical_accuracy: 0.5993 - val_sparse_top_k_categorical_accuracy: 0.8099
Epoch 9/10
281/281 [==============================] - 4s 14ms/step - loss: 0.2919 - sparse_categorical_accuracy: 0.9304 - sparse_top_k_categorical_accuracy: 0.9911 - val_loss: 2.5571 - val_sparse_categorical_accuracy: 0.6020 - val_sparse_top_k_categorical_accuracy: 0.8126
Epoch 10/10
281/281 [==============================] - 4s 14ms/step - loss: 0.2617 - sparse_categorical_accuracy: 0.9342 - sparse_top_k_categorical_accuracy: 0.9937 - val_loss: 2.6700 - val_sparse_categorical_accuracy: 0.6077 - val_sparse_top_k_categorical_accuracy: 0.8148
CPU times: user 1min 2s, sys: 8.59 s, total: 1min 10s
Wall time: 58.5 s
```


```python

```

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"算法美食屋"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![算法美食屋二维码.jpg](./data/算法美食屋二维码.jpg)
