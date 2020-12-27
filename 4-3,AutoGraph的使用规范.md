# 4-3,AutoGraph的使用规范

有三种计算图的构建方式：静态计算图，动态计算图，以及Autograph。

TensorFlow 2.0主要使用的是动态计算图和Autograph。

动态计算图易于调试，编码效率较高，但执行效率偏低。

静态计算图执行效率很高，但较难调试。

而Autograph机制可以将动态图转换成静态计算图，兼收执行效率和编码效率之利。

当然Autograph机制能够转换的代码并不是没有任何约束的，有一些编码规范需要遵循，否则可能会转换失败或者不符合预期。

我们将着重介绍Autograph的编码规范和Autograph转换成静态图的原理。

并介绍使用tf.Module来更好地构建Autograph。

本篇我们介绍使用Autograph的编码规范。

<!-- #region -->
### 一，Autograph编码规范总结


* 1，被@tf.function修饰的函数应尽可能使用TensorFlow中的函数而不是Python中的其他函数。例如使用tf.print而不是print，使用tf.range而不是range，使用tf.constant(True)而不是True.

* 2，避免在@tf.function修饰的函数内部定义tf.Variable. 

* 3，被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。

<!-- #endregion -->
```python

```

### 二，Autograph编码规范解析


 **1，被@tf.function修饰的函数应尽量使用TensorFlow中的函数而不是Python中的其他函数。**

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
#np_random每次执行都是一样的结果。
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
#tf_random每次执行都会有重新生成随机数。
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

**2，避免在@tf.function修饰的函数内部定义tf.Variable.**

```python
# 避免在@tf.function修饰的函数内部定义tf.Variable.

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

#执行将报错
#inner_var()
#inner_var()

```

```
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-12-c95a7c3c1ddd> in <module>
      7 
      8 #执行将报错
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


**3,被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等结构类型变量。**

```python
tensor_list = []

#@tf.function #加上这一行切换成Autograph结果将不符合预期！！！
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

@tf.function #加上这一行切换成Autograph结果将不符合预期！！！
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

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"算法美食屋"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![算法美食屋二维码.jpg](./data/算法美食屋二维码.jpg)
