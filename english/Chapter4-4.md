# 4-4 Mechanisms of the AutoGraph

There are three ways of constructing graph: static, dynamic and Autograph.

TensorFlow 2.X uses dynamic graph and Autograph.

Dynamic graph is easier for debugging with higher encoding efficiency, but with lower efficiency in execution.

Static graph has high efficiency in execution, but more difficult for debugging.

Autograph mechanism transforms dynamic graph into static graph, making allowance for both executing and encoding efficiencies.

There are certain rules for the code that is able to converted by Autograph, or it could result in failure or unexpected results.

We are going to introduce the coding rules of Autograph and its mechanism of converting into static graph, together with introduction about how to construct Autograph using `tf.Module`.

The coding rules of Autograph was introduced in the last section. Here we introduce the mechanisms of Autograph.



### 1. Mechanisms of Autograph


**What happens when we define a function using decorator `@tf.function` ?**

Consider the following code.

```python
import tensorflow as tf
import numpy as np 

@tf.function(autograph=True)
def myadd(a,b):
    for i in tf.range(3):
        tf.print(i)
    c = a+b
    print("tracing")
    return c
```

Nothing happens except a function signature is recorded in the stack of Python.

**What happens when this function decorated by `@tf.function` is called?**

Consider the following code.

```python
myadd(tf.constant("hello"),tf.constant("world"))
```

```
tracing
0
1
2
```

<!-- #region -->
There are two incidents:

First, the graph is created.

A static graph is created. The Python code inside this function is executed, the tensor type of each variable is determined, and the operator is added to the graph according to the order of execution. During this period, if the argument autograph=True (default) is setted, convertting of the controlling flow in Python to the one inside TensorFlow graph will happen. The majority of the work are: replacing `if` to `tf.cond` operator; replacing `while` and `for` looping to `tf.while_loop`; when necessary, add `tf.control_dependencies` to specify the dependencies of executing orders.

This is identical to the following expressions in TensorFlow 1.X:

```python
g = tf.Graph()
with g.as_default():
    a = tf.placeholder(shape=[],dtype=tf.string)
    b = tf.placeholder(shape=[],dtype=tf.string)
    cond = lambda i: i<tf.constant(3)
    def body(i):
        tf.print(i)
        return(i+1)
    loop = tf.while_loop(cond,body,loop_vars=[0])
    loop
    with tf.control_dependencies(loop):
        c = tf.strings.join([a,b])
    print("tracing")
```

The second incident is the execution of the graph.

This is identical to the following expressions in TensorFlow 1.X:

```python
with tf.Session(graph=g) as sess:
    sess.run(c,feed_dict={a:tf.constant("hello"),b:tf.constant("world")})
```

So the result for the first step comes first: A string "tracing" printed by the standard I/O stream of Python.

And next is the result of the second step: A string "1, 2, 3" printed by the standard I/O stream of TensorFlow.

<!-- #endregion -->

**What is going to happen when we call this function again with the same types of the input arguments?**

Consider the following code.

```python
myadd(tf.constant("good"),tf.constant("morning"))
```

```
0
1
2
```


Only one thing happens: execution of the graph, which is the second step mentioned above.

So the string "traicing" doesn't appear.


**What is going to happen when we call this function again with some different types of the input arguments?**

Consider the following code.

```python
myadd(tf.constant(1),tf.constant(2))
```

```
tracing
0
1
2
```


Since the data type of the argument has been changed, the previously created graph cannot be used again.

Two more tasks to be done: create new graph and execute it.

The result of the first step will be observed again, i.e. a string "tracing" printed by the standard I/O stream of Python.

And next is the result of the second step: A string "1, 2, 3" printed by the standard I/O stream of TensorFlow.


**Note: if the data type of the argument is not Tensor in the original definition of this function, then the graph will be re-created each time after calling this function.**

The demonstrated code below re-creates graph every time, so it is recommended to use Tensor type as the arguments when calling the function decorated by `@tf.function`.

```python
myadd("hello","world")
myadd("good","morning")
```

```
tracing
0
1
2
tracing
0
1
2
```

```python

```

```python

```

### 2. Scrutinize the Coding Rules of Autograph Again


We can have a better understanding to the three rules of coding of Autograph after knowing the mechanisms above.

1, We should use the TensorFlow-defined functions to be decorated by `@tf.function` as much as possible, instead of those Python functions. For instance, `tf.print` should be used instead of `print`.

Explanations: Python functions are only used during the stage of creating static graph. The Python functions are not able to be embedded into the static graph, so these Python functions are not calculated during the calling after the graph creation; in contrast, TensorFlow functions are able to be embedded into the graph. Using Python functions is causing unmatched outputs between the "eager execution" before the decoration by `@tf.function` and the "execution of static graph" after the decoration by `@tf.function`.

2，Avoid defining `tf.Variable` inside the decorator `@tf.function`.

Explanations: The defined `tf.Variable` will be re-created every time when calling the function during the "eager execution" stage. However, this re-creation of `tf.Variable` only takes place at the first step, i.e. tracing Python code to create the graph, which is introducing unmatched outputs between the "eager execution" before the decoration by `@tf.function` and the "execution of static graph" after the decoration by `@tf.function`. In fact, TensorFlow throws error in most of such cases.

3，Functions that are decorated by `@tf.function` cannot modify the variables outside the function with the data types such as Python list, dictionary, etc.

Explanations: Static graph is executed in the TensorFlow kernels, which are compiled from C++ code, thus the list and dictionary in Python are not able to be embedded into the graph. These data types can only be read during the stage of graph creating and cannot be modified during the graph execution.


```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)

```python

```
