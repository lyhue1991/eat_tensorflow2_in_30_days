# 1-3 Example: Modeling Procedure for Texts


### 1. Data Preparation


The purpose of the imdb dataset is to predict the sentiment label according to the movie reviews.

There are 20000 text reviews in the training dataset and 5000 in the testing dataset, with half positive and half negative, respectively.

The pre-processing of the text dataset is a little bit complex, which includes word division (for Chinese only, not relevant to this demonstration), dictionary construction, encoding, sequence filling, and data pipeline construction, etc.

There are two popular mothods of text preparation in TensorFlow.

The first one is constructing the text data generator using Tokenizer in `tf.keras.preprocessing`, together with `tf.keras.utils.Sequence`.

The second one is using `tf.data.Dataset`, together with the pre-processing layer `tf.keras.layers.experimental.preprocessing.TextVectorization`.

The former is more complex and is demonstrated [here](https://zhuanlan.zhihu.com/p/67697840).

The latter is the original method of TensorFlow, which is simpler.

Below is the introduction to the second method.


![](../data/电影评论.jpg)

```python
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models,layers,preprocessing,optimizers,losses,metrics
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re,string

train_data_path = "../data/imdb/train.csv"
test_data_path =  "../data/imdb/test.csv"

MAX_WORDS = 10000  # Consider the 10000 words with the highest frequency of appearance
MAX_LEN = 200  # For each sample, preserve the first 200 words
BATCH_SIZE = 20 


#Constructing data pipeline
def split_line(line):
    arr = tf.strings.split(line,"\t")
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]),tf.int32),axis = 0)
    text = tf.expand_dims(arr[1],axis = 0)
    return (text,label)

ds_train_raw =  tf.data.TextLineDataset(filenames = [train_data_path]) \
   .map(split_line,num_parallel_calls = tf.data.experimental.AUTOTUNE) \
   .shuffle(buffer_size = 1000).batch(BATCH_SIZE) \
   .prefetch(tf.data.experimental.AUTOTUNE)

ds_test_raw = tf.data.TextLineDataset(filenames = [test_data_path]) \
   .map(split_line,num_parallel_calls = tf.data.experimental.AUTOTUNE) \
   .batch(BATCH_SIZE) \
   .prefetch(tf.data.experimental.AUTOTUNE)


#Constructing dictionary
def clean_text(text):
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    cleaned_punctuation = tf.strings.regex_replace(stripped_html,
         '[%s]' % re.escape(string.punctuation),'')
    return cleaned_punctuation

vectorize_layer = TextVectorization(
    standardize=clean_text,
    split = 'whitespace',
    max_tokens=MAX_WORDS-1, #Leave one item for the placeholder
    output_mode='int',
    output_sequence_length=MAX_LEN)

ds_text = ds_train_raw.map(lambda text,label: text)
vectorize_layer.adapt(ds_text)
print(vectorize_layer.get_vocabulary()[0:100])


#Word encoding
ds_train = ds_train_raw.map(lambda text,label:(vectorize_layer(text),label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test_raw.map(lambda text,label:(vectorize_layer(text),label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)

```

```
[b'the', b'and', b'a', b'of', b'to', b'is', b'in', b'it', b'i', b'this', b'that', b'was', b'as', b'for', b'with', b'movie', b'but', b'film', b'on', b'not', b'you', b'his', b'are', b'have', b'be', b'he', b'one', b'its', b'at', b'all', b'by', b'an', b'they', b'from', b'who', b'so', b'like', b'her', b'just', b'or', b'about', b'has', b'if', b'out', b'some', b'there', b'what', b'good', b'more', b'when', b'very', b'she', b'even', b'my', b'no', b'would', b'up', b'time', b'only', b'which', b'story', b'really', b'their', b'were', b'had', b'see', b'can', b'me', b'than', b'we', b'much', b'well', b'get', b'been', b'will', b'into', b'people', b'also', b'other', b'do', b'bad', b'because', b'great', b'first', b'how', b'him', b'most', b'dont', b'made', b'then', b'them', b'films', b'movies', b'way', b'make', b'could', b'too', b'any', b'after', b'characters']
```

```python

```

### 2. Model Definition


Usually there are three ways of modeling using APIs of Keras: sequential modeling using `Sequential()` function, arbitrary modeling using functional API, and customized modeling by inheriting base class `Model`.

In this example, we use customized modeling by inheriting base class `Model`.

```python
# Actually, modeling with sequential() or API functions should be priorized.

tf.keras.backend.clear_session()

class CnnModel(models.Model):
    def __init__(self):
        super(CnnModel, self).__init__()
        
    def build(self,input_shape):
        self.embedding = layers.Embedding(MAX_WORDS,7,input_length=MAX_LEN)
        self.conv_1 = layers.Conv1D(16, kernel_size= 5,name = "conv_1",activation = "relu")
        self.pool_1 = layers.MaxPool1D(name = "pool_1")
        self.conv_2 = layers.Conv1D(128, kernel_size=2,name = "conv_2",activation = "relu")
        self.pool_2 = layers.MaxPool1D(name = "pool_2")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1,activation = "sigmoid")
        super(CnnModel,self).build(input_shape)
    
    def call(self, x):
        x = self.embedding(x)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return(x)
    
    # To show Output Shape
    def summary(self):
        x_input = layers.Input(shape = MAX_LEN)
        output = self.call(x_input)
        model = tf.keras.Model(inputs = x_input,outputs = output)
        model.summary()
    
model = CnnModel()
model.build(input_shape =(None,MAX_LEN))
model.summary()

```

```python

```

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 200)]             0         
_________________________________________________________________
embedding (Embedding)        (None, 200, 7)            70000     
_________________________________________________________________
conv_1 (Conv1D)              (None, 196, 16)           576       
_________________________________________________________________
pool_1 (MaxPooling1D)        (None, 98, 16)            0         
_________________________________________________________________
conv_2 (Conv1D)              (None, 97, 128)           4224      
_________________________________________________________________
pool_2 (MaxPooling1D)        (None, 48, 128)           0         
_________________________________________________________________
flatten (Flatten)            (None, 6144)              0         
_________________________________________________________________
dense (Dense)                (None, 1)                 6145      
=================================================================
Total params: 80,945
Trainable params: 80,945
Non-trainable params: 0
_________________________________________________________________
```

```python

```

### 3. Model Training


There are three usual ways for model training: use internal function fit, use internal function train_on_batch, and customized training loop. Here we use the customized training loop.

```python
# Time Stamp
@tf.function
def printbar():
    ts = tf.timestamp()
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

```python
optimizer = optimizers.Nadam()
loss_func = losses.BinaryCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.BinaryAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.BinaryAccuracy(name='valid_accuracy')


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
    predictions = model(features,training = False)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def train_model(model,ds_train,ds_valid,epochs):
    for epoch in tf.range(1,epochs+1):
        
        for features, labels in ds_train:
            train_step(model,features,labels)

        for features, labels in ds_valid:
            valid_step(model,features,labels)
        
        #The logs template should be modified according to metric
        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}' 
        
        if epoch%1==0:
            printbar()
            tf.print(tf.strings.format(logs,
            (epoch,train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))
            tf.print("")
        
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

train_model(model,ds_train,ds_test,epochs = 6)

```

```
================================================================================13:54:08
Epoch=1,Loss:0.442317516,Accuracy:0.7695,Valid Loss:0.323672801,Valid Accuracy:0.8614

================================================================================13:54:20
Epoch=2,Loss:0.245737702,Accuracy:0.90215,Valid Loss:0.356488883,Valid Accuracy:0.8554

================================================================================13:54:32
Epoch=3,Loss:0.17360799,Accuracy:0.93455,Valid Loss:0.361132562,Valid Accuracy:0.8674

================================================================================13:54:44
Epoch=4,Loss:0.113476314,Accuracy:0.95975,Valid Loss:0.483677238,Valid Accuracy:0.856

================================================================================13:54:57
Epoch=5,Loss:0.0698405355,Accuracy:0.9768,Valid Loss:0.607856631,Valid Accuracy:0.857

================================================================================13:55:15
Epoch=6,Loss:0.0366807655,Accuracy:0.98825,Valid Loss:0.745884955,Valid Accuracy:0.854
```


### 4. Model Evaluation


The model trained by the customized looping is not compiled, so the method `model.evaluate(ds_valid)` can not be applied directly.

```python

def evaluate_model(model,ds_valid):
    for features, labels in ds_valid:
         valid_step(model,features,labels)
    logs = 'Valid Loss:{},Valid Accuracy:{}' 
    tf.print(tf.strings.format(logs,(valid_loss.result(),valid_metric.result())))
    
    valid_loss.reset_states()
    train_metric.reset_states()
    valid_metric.reset_states()

    
```

```python
evaluate_model(model,ds_test)
```

```
Valid Loss:0.745884418,Valid Accuracy:0.854
```

```python

```

### 5. Model Application


Below are the available methods:

* model.predict(ds_test)
* model(x_test)
* model.call(x_test)
* model.predict_on_batch(x_test)

We recommend the method `model.predict(ds_test)` since it can be applied to both Dataset and Tensor.

```python
model.predict(ds_test)
```

```
array([[0.7864823 ],
       [0.9999901 ],
       [0.99944776],
       ...,
       [0.8498302 ],
       [0.13382755],
       [1.        ]], dtype=float32)
```

```python
for x_test,_ in ds_test.take(1):
    print(model(x_test))
    #Indentical expressions:
    #print(model.call(x_test))
    #print(model.predict_on_batch(x_test))
```

```
tf.Tensor(
[[7.8648227e-01]
 [9.9999011e-01]
 [9.9944776e-01]
 [3.7153201e-09]
 [9.4462049e-01]
 [2.3522753e-04]
 [1.2044354e-04]
 [9.3752089e-07]
 [9.9996352e-01]
 [9.3435925e-01]
 [9.8746723e-01]
 [9.9908626e-01]
 [4.1563155e-08]
 [4.1808244e-03]
 [8.0184749e-05]
 [8.3910513e-01]
 [3.5167937e-05]
 [7.2113985e-01]
 [4.5228912e-03]
 [9.9942589e-01]], shape=(20, 1), dtype=float32)
```

```python

```

### 6. Model Saving


Model saving with the original way of TensorFlow is recommended.

```python
model.save('../data/tf_model_savedmodel', save_format="tf")
print('export saved model.')

model_loaded = tf.keras.models.load_model('../data/tf_model_savedmodel')
model_loaded.predict(ds_test)
```

```
array([[0.7864823 ],
       [0.9999901 ],
       [0.99944776],
       ...,
       [0.8498302 ],
       [0.13382755],
       [1.        ]], dtype=float32)
```


Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)
