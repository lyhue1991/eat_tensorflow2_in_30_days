# 4-3 Rules of Using the AutoGraph

There are three ways of constructing graph: static, dynamic and Autograph.

TensorFlow 2.X uses dynamic graph and Autograph.

Dynamic graph is easier for debugging with higher encoding efficiency, but with lower efficiency in execution.

Static graph has high efficiency in execution, but more difficult for debugging.

Autograph mechanism transforms dynamic graph into static graph, making allowance for both executing and encoding efficiencies.

There are certain rules for the code that is able to converted by Autograph, or it could result in failure or unexpected results.

We are going to introduce the coding rules of Autograph and its mechanism of converting into static graph, together with introduction about how to construct Autograph using `tf.Module`.

This section introduce the coding rules of using Autograph. We will introduce the mechanisms of Autograph in next section and explain the logic behind the rules there.

<!-- #region -->
### 1. Summarization of the Coding Rules of Autograph


* 1. We should use the TensorFlow-defined functions to be decorated by `@tf.function` as much as possible, instead of those Python functions. For instance, `tf.print` should be used instead of `print`; `tf.range` should be used instead of `range`; `tf.constant(True)` should be used instead of `True`.

* 2. Avoid defining `tf.Variable` inside the decorator `@tf.function`.

* 3. Functions that are decorated by `@tf.function` cannot modify the struct data types variables outside the function such as Python list, dictionary, etc.

<!-- #endregion -->
```python

```

### 2. Explanations to the Autograph Coding Rules


 **2.1  We should use the TensorFlow-defined functions to be decorated by `@tf.function` as much as possible, instead of those Python functions.**

```python
import numpy as np
import tensorflow as tf

@tf.function
def np_random():
    a = np.random.randn(3,3)
    tf.print(a)
    
@tf.function
def tf_random():
    a = tf.random.normal((3,3))
    tf.print(a)
```

```python
# Same results after each execution of np_random
np_random()
np_random()
```

```
array([[ 0.22619201, -0.4550123 , -0.42587565],
       [ 0.05429906,  0.2312667 , -1.44819738],
       [ 0.36571796,  1.45578986, -1.05348983]])
array([[ 0.22619201, -0.4550123 , -0.42587565],
       [ 0.05429906,  0.2312667 , -1.44819738],
       [ 0.36571796,  1.45578986, -1.05348983]])
```

```python
# New random numbers are generated after each execution of tf_random
tf_random()
tf_random()
```

```
[[-1.38956189 -0.394843668 0.420657277]
 [2.87235498 -1.33740318 -0.533843279]
 [0.918233037 0.118598573 -0.399486482]]
[[-0.858178258 1.67509317 0.511889517]
 [-0.545829177 -2.20118237 -0.968222201]
 [0.733958483 -0.61904633 0.77440238]]
```

```python

```

**2.2 Avoid defining `tf.Variable` inside the decorator `@tf.function`.**

```python
# Avoid defining tf.Variable inside the decorator @tf.function.

x = tf.Variable(1.0,dtype=tf.float32)
@tf.function
def outer_var():
    x.assign_add(1.0)
    tf.print(x)
    return(x)

outer_var() 
outer_var()

```

```python
@tf.function
def inner_var():
    x = tf.Variable(1.0,dtype = tf.float32)
    x.assign_add(1.0)
    tf.print(x)
    return(x)

# Error after execution
#inner_var()
#inner_var()

```

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-12-c95a7c3c1ddd> in <module>
      7 
      8 # Error after execution
----> 9 inner_var()
     10 inner_var()

~/anaconda3/lib/python3.7/site-packages/tensorflow_core/python/eager/def_function.py in __call__(self, *args, **kwds)
    566         xla_context.Exit()
    567     else:
--> 568       result = self._call(*args, **kwds)
    569 
    570     if tracing_count == self._get_tracing_count():
......
ValueError: tf.function-decorated function tried to create variables on non-first call.
```


**2.3  Functions that are decorated by `@tf.function` cannot modify the struct data types variables outside the function such as Python list, dictionary, etc.**

```python
tensor_list = []

#@tf.function # Autograph will result in something unexpected if executing this line
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)

```

```
[<tf.Tensor: shape=(), dtype=float32, numpy=5.0>, <tf.Tensor: shape=(), dtype=float32, numpy=6.0>]
```

```python
tensor_list = []

@tf.function # Autograph will result in something unexpected if executing this line
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list


append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)

```

```
[<tf.Tensor 'x:0' shape=() dtype=float32>]
```

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.


![image.png](../data/Python与算法之美logo.jpg)
