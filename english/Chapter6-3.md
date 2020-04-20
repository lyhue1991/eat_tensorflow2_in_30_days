# 6-3 Model Training Using Single GPU

The training procedure of deep learning is usually time consuming. It even takes tens of days for training, and there is no need to mention those take days or hours.

The time is mainly consumpted by two stages, data preparation and parameter iteration.

We can increase the number of process to alleviate this issue if data preparation takes the majority of time.

However, if the majority of time is taken by parameter iteration, we need to use GPU or Google TPU for acceleration.

You may refer to this article for further details: ["GPU acceleration for Keras Models - How to Use Free Colab GPUs (in Chinese)"](https://zhuanlan.zhihu.com/p/68509398)

There is no need to modify source code for switching from CPU to GPU when using the pre-defined `fit` method or the customized training loops. When GPU is available and the device is not specified, TensorFlow automatically chooses GPU for tensor creating and computing.

However, for the case of using shared GPU with multiple users, sucha as using server of the company or the lab, we need to add following code to specify the GPU ID and the GPU memory size that we are going to use, in order to avoid the GPU resources to be occupied by a single user (actually TensorFlow acquires all GPU cors and all GPU memories by default) and allows multiple users perform training on it.


In Colab notebook, choose GPU in Edit -> Notebook Settings -> Hardware Accelerator

Note: the following code only executes on Colab.

You may use the following link for testing (tf_singleGPU, in Chinese)

https://colab.research.google.com/drive/1r5dLoeJq5z01sU72BX2M5UiNSkuxsEFe

```python
%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)
```

```python
from tensorflow.keras import * 

# Time stamp
@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts%(24*60*60)

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
    tf.print("=========="*8,end = "")
    tf.print(timestring)
    
```

### 1. GPU Configuration

```python
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    gpu0 = gpus[0] # Only use GPU 0 when existing multiple GPUs
    tf.config.experimental.set_memory_growth(gpu0, True) # Set the usage of GPU memory according to needs
    # The GPU memory usage could also be fixed (e.g. 4GB)
    #tf.config.experimental.set_virtual_device_configuration(gpu0,
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) 
    tf.config.set_visible_devices([gpu0],"GPU") 
```

Compare the computing speed between GPU and CPU.

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

### 2. Data Preparation

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

### 3. Model Defining

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

### 4. Model Training

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

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)
