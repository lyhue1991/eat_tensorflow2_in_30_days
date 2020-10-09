# 6-6 Model Deploying Using tensorflow-serving

There are multiple ways to deploy and run the trained models which saved with the original tensorflow format. 

For example:

We can load and run the model in the web browser using javascript through `tensorflow-js`.

We can load and run the TensorFlow model on mobile and embeded devices through `tensorflow-lite`.

We can use `tensorflow-serving` to load the model that providing network interface API service and to acquire the prediction results from the model through sending network requests in arbitrary programming languages.

We can predict using the TensorFlow model in Java or spark (scala) through the `TensorFlow for Java` port.

This section introduces model deploying by `tensorflow serving` and using spark (scala) to implement the TensorFlow models.

```python

```

### 0. Introduction to model deploying by tensorflow serving

<!-- #region -->
The necessary steps of model deploying using tensorflow serving are:

* (1) Prepare the protobuf model file.

* (2) Install the tensorflow serving.

* (3) Start the tensorflow serving service.

* (4) Send the request to the API service to obtain the prediction.


You may use the following link for testing (tf_serving, in Chinese)
https://colab.research.google.com/drive/1vS5LAYJTEn-H0GDb1irzIuyRB8E3eWc8

<!-- #endregion -->

```python
%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import * 

```

### 1. Prepare the protobuf Model File

Here we train a simple linear regression model with `tf.keras` and save it as protobuf file.

```python
import tensorflow as tf
from tensorflow.keras import models,layers,optimizers

## Number of samples
n = 800

## Generating testing dataset
X = tf.random.uniform([n,2],minval=-10,maxval=10) 
w0 = tf.constant([[2.0],[-1.0]])
b0 = tf.constant(3.0)

Y = X@w0 + b0 + tf.random.normal([n,1],
    mean = 0.0,stddev= 2.0) # @ is matrix multiplication; adding Gaussian noise

## Modeling
tf.keras.backend.clear_session()
inputs = layers.Input(shape = (2,),name ="inputs") # Set the input name as "inputs"
outputs = layers.Dense(1, name = "outputs")(inputs) # Set the output name as "outputs"
linear = models.Model(inputs = inputs,outputs = outputs)
linear.summary()

## Training with fit method
linear.compile(optimizer="rmsprop",loss="mse",metrics=["mae"])
linear.fit(X,Y,batch_size = 8,epochs = 100)  

tf.print("w = ",linear.layers[1].kernel)
tf.print("b = ",linear.layers[1].bias)

## Save the model as pb format
export_path = "../data/linear_model/"
version = "1"       # Version could be used for management of further updates
linear.save(export_path+version, save_format="tf") 
```

```python
# Check the saved model file
!ls {export_path+version}
```

```
assets	saved_model.pb	variables
```

```python
# Check the info of the model file
!saved_model_cli show --dir {export_path+str(version)} --all
```

```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 2)
        name: serving_default_inputs:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['outputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict
WARNING:tensorflow:From /tensorflow-2.1.0/python3.6/tensorflow_core/python/ops/resource_variable_ops.py:1786: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None

  Function Name: '_default_save_signature'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')

  Function Name: 'call_and_return_all_conditional_losses'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
```

```python

```

### 2. Installing tensorflow serving


Two methods for installing tensorflow serving: Using Docker images, or using apt.

Docker image is the simplest way of installation and we recommend it.

Docker is a container that provides independent environment for various programs.

The companies that are using TensorFlow usually use Docker to install tensorflow serving by operation experts, so the algorithm engineers don't have to worry about the installation.

The installation of Docker on different OS are shown below (in Chinese).

Windows: https://www.runoob.com/docker/windows-docker-install.html

MacOs: https://www.runoob.com/docker/macos-docker-install.html

CentOS: https://www.runoob.com/docker/centos-docker-install.html

After successful installation of Docker, run the following command to load the tensorflow/serving image.

`docker pull tensorflow/serving`


```python

```

### 3. Starting tensorflow serving Service

```python
!docker run -t --rm -p 8501:8501 \
    -v "/Users/.../data/linear_model/" \
    -e MODEL_NAME=linear_model \
    tensorflow/serving & >server.log 2>&1
```

```python

```

### 4. Sending request to the API service


The request could be sent through http function in any kind of the programming languages. We demonstrate request sending using the `curl` command in Linux and the `requests` library in Python.

```python
!curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/linear_model:predict
```

```
{
    "predictions": [[3.06546211], [5.01313448]
    ]
}
```

```python
import json,requests

data = json.dumps({"signature_name": "serving_default", "instances": [[1.0, 2.0], [5.0,7.0]]})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/linear_model:predict', 
        data=data, headers=headers)
predictions = json.loads(json_response.text)["predictions"]
print(predictions)
```

```
[[3.06546211], [6.02843142]]
```

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)

```python

```

```python

```
