# 2-1 Data Structure

Program = Data Structure + Algorithm

TensorFlow Program = Data Structure of Tensor + Algorithm in Graph

Tensor and graph are key concepts of TensorFlow.

The fundamental data structure in TensorFlow is Tensor, which is multi-dimentional array. Tensor is similar with the `array` in numpy.

There are two types of tensor accoring to the behavior: constant and variable.

The value of constant cannot be re-assigned in the graph, while variable can be re-assigned through operators such as `assign`.


### 1. Constant Tensor


The data type of tensor is basically corresponding to `numpy.array`.

```python
import numpy as np
import tensorflow as tf

i = tf.constant(1) # tf.int32 type constant
l = tf.constant(1,dtype = tf.int64) # tf.int64 type constant
f = tf.constant(1.23) #tf.float32 type constant
d = tf.constant(3.14,dtype = tf.double) # tf.double type constant
s = tf.constant("hello world") # tf.string type constant
b = tf.constant(True) #tf.bool type constant


print(tf.int64 == np.int64) 
print(tf.bool == np.bool)
print(tf.double == np.float64)
print(tf.string == np.unicode) # tf.string type is not equal to np.unicode type

```

```
True
True
True
False
```


Each data type can be represented by tensor in different rank.

Scalars are tensors with rank = 0, arrays are with rank = 1, matrix are with rank = 2

Colorful image has three channels (RGB), which can be represented as a tensor with rank = 3.

There is a temporal dimension for video so it could be represented as a rank 4 tensor.

An intuitive way to understand: the number of the square brackets equals to the rank of the tensor.

```python
scalar = tf.constant(True)  #A scalar is a rank 0 tensor

print(tf.rank(scalar))
print(scalar.numpy().ndim)  # tf.rank equals to the ndim function in numpy
```

```
tf.Tensor(0, shape=(), dtype=int32)
0
```

```python
vector = tf.constant([1.0,2.0,3.0,4.0]) #A vector is a rank 1 tensor

print(tf.rank(vector))
print(np.ndim(vector.numpy()))
```

```
tf.Tensor(1, shape=(), dtype=int32)
1
```

```python
matrix = tf.constant([[1.0,2.0],[3.0,4.0]]) #A matrix is a rank 2 tensor

print(tf.rank(matrix).numpy())
print(np.ndim(matrix))
```

```
2
2
```

```python
tensor3 = tf.constant([[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]])  # A rank 3 tensor
print(tensor3)
print(tf.rank(tensor3))
```

```
tf.Tensor(
[[[1. 2.]
  [3. 4.]]

 [[5. 6.]
  [7. 8.]]], shape=(2, 2, 2), dtype=float32)
tf.Tensor(3, shape=(), dtype=int32)
```

```python
tensor4 = tf.constant([[[[1.0,1.0],[2.0,2.0]],[[3.0,3.0],[4.0,4.0]]],
                        [[[5.0,5.0],[6.0,6.0]],[[7.0,7.0],[8.0,8.0]]]])  # A rank 4 tensor
print(tensor4)
print(tf.rank(tensor4))
```

```
tf.Tensor(
[[[[1. 1.]
   [2. 2.]]

  [[3. 3.]
   [4. 4.]]]


 [[[5. 5.]
   [6. 6.]]

  [[7. 7.]
   [8. 8.]]]], shape=(2, 2, 2, 2), dtype=float32)
tf.Tensor(4, shape=(), dtype=int32)
```


We use tf.cast to change the data type of the tensors.

The method `numpy()` is for converting the data type from tensor to numpy array.

The method `shape` is for checking up the size of tensor.

```python
h = tf.constant([123,456],dtype = tf.int32)
f = tf.cast(h,tf.float32)
print(h.dtype, f.dtype)
```

```
<dtype: 'int32'> <dtype: 'float32'>
```

```python
y = tf.constant([[1.0,2.0],[3.0,4.0]])
print(y.numpy()) #Convert to np.array
print(y.shape)
```

```
[[1. 2.]
 [3. 4.]]
(2, 2)
```

```python
u = tf.constant(u"Hello World")
print(u.numpy())  
print(u.numpy().decode("utf-8"))
```

```
b'\xe4\xbd\xa0\xe5\xa5\xbd \xe4\xb8\x96\xe7\x95\x8c'
Hello World
```

```python

```
### 2. Variable Tensor


The trainable parameters in the models are usually defined as variables.

```python
# The value of a constant is NOT changeable. Re-assignment creates a new space in the memory.
c = tf.constant([1.0,2.0])
print(c)
print(id(c))
c = c + tf.constant([1.0,1.0])
print(c)
print(id(c))
```

```
tf.Tensor([1. 2.], shape=(2,), dtype=float32)
5276289568
tf.Tensor([2. 3.], shape=(2,), dtype=float32)
5276290240
```

```python
# The value of a variable is changeable through re-assigning methods such as assign, assign_add, etc.
v = tf.Variable([1.0,2.0],name = "v")
print(v)
print(id(v))
v.assign_add([1.0,1.0])
print(v)
print(id(v))
```
```
<tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([1., 2.], dtype=float32)>
5276259888
<tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([2., 3.], dtype=float32)>
5276259888

```

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)


