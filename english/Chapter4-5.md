# 4-5 AutoGraph and tf.Module


There are three ways of constructing graph: static, dynamic and Autograph.

TensorFlow 2.X uses dynamic graph and Autograph.

Dynamic graph is easier for debugging with higher encoding efficiency, but with lower efficiency in execution.

Static graph has high efficiency in execution, but more difficult for debugging.

Autograph mechanism transforms dynamic graph into static graph, making allowance for both executing and encoding efficiencies.

There are certain rules for the code that is able to converted by Autograph, or it could result in failure or unexpected results.

The coding rules and the mechanisms of Autograph were introduced in the last sections.

In this section, we introduce constructing Autograph using `tf.Module`.



### 1. Introduction to Autograph and `tf.Module`


We mentioned that the definition of `tf.Variable` should be avoided inside the decorator `@tf.function`.

However, it would seem to be a imperfect leaked package if we define `tf.Variable` outside the function, since the function has outside dependency.

One of the simple solutions is: defining a class and place the definition of `tf.Variable` inside the initial method, and leave the other methods/implementation elsewhere.

After such an ingenious operation, it is so naturally as if the chronic constipation was cured by the best laxative.

The surprise is that TensorFlow providing us a base class `tf.Module` to get the above naturally. What's more, It is supposed to be inherited for constructing child classes to manage variables and other `Module` conveniently. And the most important that it allows us to save model through `tf.saved_model` and achieve cross-platform deployment. What a surprise!

In fact, `tf.keras.models.Model`, `tf.keras.layers.Layer` are both inherited from `tf.Module`. They provides the management to the variables and the referred sub-modules.

**We are able to develop arbitrary learning model (not only neural network) and implement cross-platform deployment through the packaging provided by `tf.Module` and the low-level APIs in TensorFlow.**


### 2. Packaging Autograph Using `tf.Module`


We define a simple function。

```python
import tensorflow as tf 
x = tf.Variable(1.0,dtype=tf.float32)

# Use input_signature to limit the signature type of the input tensors with shape and dtype inside the decorator tf.function
@tf.function(input_signature=[tf.TensorSpec(shape = [], dtype = tf.float32)])    
def add_print(a):
    x.assign_add(a)
    tf.print(x)
    return(x)
```

```python
add_print(tf.constant(3.0))
#add_print(tf.constant(3)) # Error: argument inconsistent with the tensor signature.
```

```
4
```


Package using `tf.Module`.

```python
class DemoModule(tf.Module):
    def __init__(self,init_value = tf.constant(0.0),name=None):
        super(DemoModule, self).__init__(name=name)
        with self.name_scope:  # Identical to: with tf.name_scope("demo_module")
            self.x = tf.Variable(init_value,dtype = tf.float32,trainable=True)

     
    @tf.function(input_signature=[tf.TensorSpec(shape = [], dtype = tf.float32)])  
    def addprint(self,a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return(self.x)

```

```python
# Execute
demo = DemoModule(init_value = tf.constant(1.0))
result = demo.addprint(tf.constant(5.0))
```

```
6
```

```python
# Browse all variables and trainable variables in the module
print(demo.variables)
print(demo.trainable_variables)
```

```
(<tf.Variable 'demo_module/Variable:0' shape=() dtype=float32, numpy=6.0>,)
(<tf.Variable 'demo_module/Variable:0' shape=() dtype=float32, numpy=6.0>,)
```

```python
# Browse all sub-modules
demo.submodules
```

```python
# Save the model using tf.saved_model and specify the method of cross-platform deployment.
tf.saved_model.save(demo,"../data/demo/1",signatures = {"serving_default":demo.addprint})
```

```python
# Load the modle
demo2 = tf.saved_model.load("../data/demo/1")
demo2.addprint(tf.constant(5.0))
```

```
11
```

```python
# Check the info of the model file. The info in the red rectangulars could be used during the deployment and the cross-platform usage.
!saved_model_cli show --dir ../data/demo/1 --all
```

![](../data/查看模型文件信息.jpg)

```python

```

Check the graph in tensorboard, the module will be added with name `demo_module`, showing the hierarchy of the graph.

```python
import datetime

# Creating log
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = '../data/demomodule/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

# Start tracing of the Autograph
tf.summary.trace_on(graph=True, profiler=True) 

# Execute the Autograph
demo = DemoModule(init_value = tf.constant(0.0))
result = demo.addprint(tf.constant(5.0))

# Write the info of the graph into the log
with writer.as_default():
    tf.summary.trace_export(
        name="demomodule",
        step=0,
        profiler_outdir=logdir)
    
```

```python

```

```python
# Magic command to launch tensorboard in jupyter
%reload_ext tensorboard
```

```python
from tensorboard import notebook
notebook.list() 
```

```python
notebook.start("--logdir ../data/demomodule/")
```

![](../data/demomodule的计算图结构.jpg)

```python

```

Besides using the child class of `tf.Module`, it is also possible to package through adding attribute to `tf.Module`.

```python
mymodule = tf.Module()
mymodule.x = tf.Variable(0.0)

@tf.function(input_signature=[tf.TensorSpec(shape = [], dtype = tf.float32)])  
def addprint(a):
    mymodule.x.assign_add(a)
    tf.print(mymodule.x)
    return (mymodule.x)

mymodule.addprint = addprint
```

```python
mymodule.addprint(tf.constant(1.0)).numpy()
```

```
1.0
```

```python
print(mymodule.variables)
```

```
(<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.0>,)
```

```python
# Save model using tf.saved_model
tf.saved_model.save(mymodule,"../data/mymodule",
    signatures = {"serving_default":mymodule.addprint})

# Load the model
mymodule2 = tf.saved_model.load("../data/mymodule")
mymodule2.addprint(tf.constant(5.0))
```

```
INFO:tensorflow:Assets written to: ../data/mymodule/assets
5
```

```python

```

### 3. `tf.Module` and `tf.keras.Model`，`tf.keras.layers.Layer`


The models and the layers in `tf.keras` are implemented through inheriting `tf.Module`. Both of them are able to manage variables and sub-modules.

```python
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics
```

```python
print(issubclass(tf.keras.Model,tf.Module))
print(issubclass(tf.keras.layers.Layer,tf.Module))
print(issubclass(tf.keras.Model,tf.keras.layers.Layer))
```

```
True
True
True
```

```python
tf.keras.backend.clear_session() 

model = models.Sequential()

model.add(layers.Dense(4,input_shape = (10,)))
model.add(layers.Dense(2))
model.add(layers.Dense(1))
model.summary()
```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 4)                 44        
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 10        
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3         
=================================================================
Total params: 57
Trainable params: 57
Non-trainable params: 0
_________________________________________________________________
```

```python
model.variables
```

```
[<tf.Variable 'dense/kernel:0' shape=(10, 4) dtype=float32, numpy=
 array([[-0.06741005,  0.45534766,  0.5190817 , -0.01806331],
        [-0.14258742, -0.49711505,  0.26030976,  0.18607801],
        [-0.62806034,  0.5327399 ,  0.42206633,  0.29201728],
        [-0.16602087, -0.18901917,  0.55159235, -0.01091868],
        [ 0.04533798,  0.326845  , -0.582667  ,  0.19431782],
        [ 0.6494713 , -0.16174704,  0.4062966 ,  0.48760796],
        [ 0.58400524, -0.6280886 , -0.11265379, -0.6438277 ],
        [ 0.26642334,  0.49275804,  0.20793378, -0.43889117],
        [ 0.4092741 ,  0.09871006, -0.2073121 ,  0.26047975],
        [ 0.43910992,  0.00199282, -0.07711256, -0.27966842]],
       dtype=float32)>,
 <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>,
 <tf.Variable 'dense_1/kernel:0' shape=(4, 2) dtype=float32, numpy=
 array([[ 0.5022683 , -0.0507431 ],
        [-0.61540484,  0.9369011 ],
        [-0.14412141, -0.54607415],
        [ 0.2027781 , -0.4651153 ]], dtype=float32)>,
 <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
 <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
 array([[-0.244825 ],
        [-1.2101456]], dtype=float32)>,
 <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
```

```python
model.layers[0].trainable = False # Freeze the variables in layer 0, make it untrainable.
model.trainable_variables
```

```
[<tf.Variable 'dense_1/kernel:0' shape=(4, 2) dtype=float32, numpy=
 array([[ 0.5022683 , -0.0507431 ],
        [-0.61540484,  0.9369011 ],
        [-0.14412141, -0.54607415],
        [ 0.2027781 , -0.4651153 ]], dtype=float32)>,
 <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
 <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
 array([[-0.244825 ],
        [-1.2101456]], dtype=float32)>,
 <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
```

```python
model.submodules
```

```
(<tensorflow.python.keras.engine.input_layer.InputLayer at 0x144d8c080>,
 <tensorflow.python.keras.layers.core.Dense at 0x144daada0>,
 <tensorflow.python.keras.layers.core.Dense at 0x144d8c5c0>,
 <tensorflow.python.keras.layers.core.Dense at 0x144d7aa20>)
```

```python
model.layers
```

```
[<tensorflow.python.keras.layers.core.Dense at 0x144daada0>,
 <tensorflow.python.keras.layers.core.Dense at 0x144d8c5c0>,
 <tensorflow.python.keras.layers.core.Dense at 0x144d7aa20>]
```

```python
print(model.name)
print(model.name_scope())
```

```
sequential
sequential
```

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)
