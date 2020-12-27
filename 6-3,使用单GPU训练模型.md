# 6-3,使用单GPU训练模型

深度学习的训练过程常常非常耗时，一个模型训练几个小时是家常便饭，训练几天也是常有的事情，有时候甚至要训练几十天。

训练过程的耗时主要来自于两个部分，一部分来自数据准备，另一部分来自参数迭代。

当数据准备过程还是模型训练时间的主要瓶颈时，我们可以使用更多进程来准备数据。

当参数迭代过程成为训练时间的主要瓶颈时，我们通常的方法是应用GPU或者Google的TPU来进行加速。

详见《用GPU加速Keras模型——Colab免费GPU使用攻略》

https://zhuanlan.zhihu.com/p/68509398


无论是内置fit方法，还是自定义训练循环，从CPU切换成单GPU训练模型都是非常方便的，无需更改任何代码。当存在可用的GPU时，如果不特意指定device，tensorflow会自动优先选择使用GPU来创建张量和执行张量计算。

但如果是在公司或者学校实验室的服务器环境，存在多个GPU和多个使用者时，为了不让单个同学的任务占用全部GPU资源导致其他同学无法使用（tensorflow默认获取全部GPU的全部内存资源权限，但实际上只使用一个GPU的部分资源），我们通常会在开头增加以下几行代码以控制每个任务使用的GPU编号和显存大小，以便其他同学也能够同时训练模型。


在Colab笔记本中：修改->笔记本设置->硬件加速器 中选择 GPU

注：以下代码只能在Colab 上才能正确执行。

可通过以下colab链接测试效果《tf_单GPU》：

https://colab.research.google.com/drive/1r5dLoeJq5z01sU72BX2M5UiNSkuxsEFe

```python
%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)
```

```python
from tensorflow.keras import * 

#打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp()%(24*60*60)

    hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)
    
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))
    
    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=========="*8+timestring)
    
```

### 一，GPU设置

```python
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
    # 或者也可以设置GPU显存为固定使用量(例如：4G)
    #tf.config.experimental.set_virtual_device_configuration(gpu0,
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) 
    tf.config.set_visible_devices([gpu0],"GPU") 
```

比较GPU和CPU的计算速度

```python
printbar()
with tf.device("/gpu:0"):
    tf.random.set_seed(0)
    a = tf.random.uniform((10000,100),minval = 0,maxval = 3.0)
    b = tf.random.uniform((100,100000),minval = 0,maxval = 3.0)
    c = a@b
    tf.print(tf.reduce_sum(tf.reduce_sum(c,axis = 0),axis=0))
printbar()
```

```
================================================================================17:37:01
2.24953778e+11
================================================================================17:37:01
```

```python
printbar()
with tf.device("/cpu:0"):
    tf.random.set_seed(0)
    a = tf.random.uniform((10000,100),minval = 0,maxval = 3.0)
    b = tf.random.uniform((100,100000),minval = 0,maxval = 3.0)
    c = a@b
    tf.print(tf.reduce_sum(tf.reduce_sum(c,axis = 0),axis=0))
printbar()
```

```
================================================================================17:37:34
2.24953795e+11
================================================================================17:37:40
```

```python

```

### 二，准备数据

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

```python

```

### 三，定义模型

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

model = create_model()
model.summary()

```

```
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

```

### 四，训练模型

```python
optimizer = optimizers.Nadam()
loss_func = losses.SparseCategoricalCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.SparseCategoricalAccuracy(name='valid_accuracy')

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features,training = True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)
    
@tf.function
def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)
    

def train_model(model,ds_train,ds_valid,epochs):
    for epoch in tf.range(1,epochs+1):
        
        for features, labels in ds_train:
            train_step(model,features,labels)

        for features, labels in ds_valid:
            valid_step(model,features,labels)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
        
        if epoch%1 ==0:
            printbar()
            tf.print(tf.strings.format(logs,
            (epoch,train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))
            tf.print("")
            
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

train_model(model,ds_train,ds_test,10)
```

```python

```

```
================================================================================17:13:26
Epoch=1,Loss:1.96735072,Accuracy:0.489200622,Valid Loss:1.64124215,Valid Accuracy:0.582813919

================================================================================17:13:28
Epoch=2,Loss:1.4640888,Accuracy:0.624805152,Valid Loss:1.5559175,Valid Accuracy:0.607747078

================================================================================17:13:30
Epoch=3,Loss:1.20681274,Accuracy:0.68581605,Valid Loss:1.58494771,Valid Accuracy:0.622439921

================================================================================17:13:31
Epoch=4,Loss:0.937500894,Accuracy:0.75361836,Valid Loss:1.77466083,Valid Accuracy:0.621994674

================================================================================17:13:33
Epoch=5,Loss:0.693960547,Accuracy:0.822199941,Valid Loss:2.00267363,Valid Accuracy:0.6197685

================================================================================17:13:35
Epoch=6,Loss:0.519614,Accuracy:0.870296121,Valid Loss:2.23463202,Valid Accuracy:0.613980412

================================================================================17:13:37
Epoch=7,Loss:0.408562034,Accuracy:0.901246965,Valid Loss:2.46969271,Valid Accuracy:0.612199485

================================================================================17:13:39
Epoch=8,Loss:0.339028627,Accuracy:0.920062363,Valid Loss:2.68585229,Valid Accuracy:0.615316093

================================================================================17:13:41
Epoch=9,Loss:0.293798745,Accuracy:0.92930305,Valid Loss:2.88995624,Valid Accuracy:0.613535166

================================================================================17:13:43
Epoch=10,Loss:0.263130337,Accuracy:0.936651051,Valid Loss:3.09705234,Valid Accuracy:0.612644672
```

```python

```

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"算法美食屋"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![算法美食屋二维码.jpg](./data/算法美食屋二维码.jpg)
