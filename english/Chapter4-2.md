# 4-2 Mathematical Operations of the Tensor

Tensor operation includes structural operation and mathematical operation.

The structural operation includes tensor creation, index slicing, dimension transform, combining & splitting, etc.

The mathematical operation includes scalar operation, vector operation, and matrix operation. We will also introduce the broadcasting mechanism of tensor operation.

This section is about the mathematical operation of tensor.

```python

```

### 1. Scalar Operation


The mathematical operation includes scalar operation, vector operation, and matrix operation.

The scalar operation includes add, subtract, multiply, divide, power, and trigonometric functions, exponential functions, log functions, and logical comparison, etc.

The scalar operation is an element-by-element operation.

Some of the scalar operators are overloaded from the normal mathematical operators and support broadcasting similar as numpy.

Most scalar operators are under the module `tf.math`.

```python
import tensorflow as tf 
import numpy as np 
```

```python
a = tf.constant([[1.0,2],[-3,4.0]])
b = tf.constant([[5.0,6],[7.0,8.0]])
a+b  # Operator overloading
```

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 6.,  8.],
       [ 4., 12.]], dtype=float32)>
```

```python
a-b 
```

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ -4.,  -4.],
       [-10.,  -4.]], dtype=float32)>
```

```python
a*b 
```

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[  5.,  12.],
       [-21.,  32.]], dtype=float32)>
```

```python
a/b
```

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 0.2       ,  0.33333334],
       [-0.42857143,  0.5       ]], dtype=float32)>
```

```python
a**2
```

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 1.,  4.],
       [ 9., 16.]], dtype=float32)>
```

```python
a**(0.5)
```

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[1.       , 1.4142135],
       [      nan, 2.       ]], dtype=float32)>
```

```python
a%3 # Reloading of mod operator, identical to: m = tf.math.mod(a,3)
```

```
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 0], dtype=int32)>
```

```python
a//3  # Divid and round towards negative infinity
```

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 0.,  0.],
       [-1.,  1.]], dtype=float32)>
```

```python
(a>=2)
```

```
<tf.Tensor: shape=(2, 2), dtype=bool, numpy=
array([[False,  True],
       [False,  True]])>
```

```python
(a>=2)&(a<=3)
```

```
<tf.Tensor: shape=(2, 2), dtype=bool, numpy=
array([[False,  True],
       [False, False]])>
```

```python
(a>=2)|(a<=3)
```

```
<tf.Tensor: shape=(2, 2), dtype=bool, numpy=
array([[ True,  True],
       [ True,  True]])>
```

```python
a==5 #tf.equal(a,5)
```

```
<tf.Tensor: shape=(3,), dtype=bool, numpy=array([False, False, False])>
```

```python
tf.sqrt(a)
```

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[1.       , 1.4142135],
       [      nan, 2.       ]], dtype=float32)>
```

```python
a = tf.constant([1.0,8.0])
b = tf.constant([5.0,6.0])
c = tf.constant([6.0,7.0])
tf.add_n([a,b,c])
```

```
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([12., 21.], dtype=float32)>
```

```python
tf.print(tf.maximum(a,b))
```

```
[5 8]
```

```python
tf.print(tf.minimum(a,b))
```

```
[1 6]
```

```python
# clip value
x = tf.constant([0.9,-0.8,100.0,-20.0,0.7])
y = tf.clip_by_value(x,clip_value_min=-1,clip_value_max=1)
z = tf.clip_by_norm(x,clip_norm = 3)
tf.print(y)
tf.print(z)
```

```
[0.9 -0.8 1 -1 0.7]
[0.0264732055 -0.0235317405 2.94146752 -0.588293493 0.0205902718]
```


### 2. Vector Operation


Vector operation manipulate along one specific axis. It projects one vector to a scalar or another vector.
Many names of vector operator starts with "reduce".

```python
# Vector "reduce"
a = tf.range(1,10)
tf.print(tf.reduce_sum(a))
tf.print(tf.reduce_mean(a))
tf.print(tf.reduce_max(a))
tf.print(tf.reduce_min(a))
tf.print(tf.reduce_prod(a))
```

```
45
5
9
1
362880
```

```python
# "reduce" along the specific dimension
b = tf.reshape(a,(3,3))
tf.print(tf.reduce_sum(b, axis=1, keepdims=True))
tf.print(tf.reduce_sum(b, axis=0, keepdims=True))
```

```
[[6]
 [15]
 [24]]
[[12 15 18]]
```

```python
# "reduce" for bool type
p = tf.constant([True,False,False])
q = tf.constant([False,False,True])
tf.print(tf.reduce_all(p))
tf.print(tf.reduce_any(q))
```

```
0
1
```

```python
# Implement tf.reduce_sum using tf.foldr
s = tf.foldr(lambda a,b:a+b,tf.range(10)) 
tf.print(s)
```

```
45
```

```python
# Cumulative sum
a = tf.range(1,10)
tf.print(tf.math.cumsum(a))
tf.print(tf.math.cumprod(a))
```

```
[1 3 6 ... 28 36 45]
[1 2 6 ... 5040 40320 362880]
```

```python
# Index of max and min values in the arguments
a = tf.range(1,10)
tf.print(tf.argmax(a))
tf.print(tf.argmin(a))
```

```
8
0
```

```python
# Sort the elements in the tensor using tf.math.top_k
a = tf.constant([1,3,7,5,4,8])

values,indices = tf.math.top_k(a,3,sorted=True)
tf.print(values)
tf.print(indices)

# tf.math.top_k is able to implement KNN algorithm in TensorFlow
```

```
[8 7 5]
[5 2 3]
```

```python

```

### 3. Matrix Operation


Matrix must be two-dimensional. Something such as `tf.constant([1,2,3])` is not a matrix.

Matrix operation includes matrix multiply, transpose, inverse, trace, norm, determinant, eigenvalue, decomposition, etc.

Most of the matrix operations are in the `tf.linalg` except for some popular operations.

```python
# Matrix multiplication
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[2,0],[0,2]])
a@b  # Identical to tf.matmul(a,b)
```

```
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[2, 4],
       [6, 8]], dtype=int32)>
```

```python
# Matrix transpose
a = tf.constant([[1.0,2],[3,4]])
tf.transpose(a)
```

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[1., 3.],
       [2., 4.]], dtype=float32)>
```

```python
# Matrix inverse, must be in type of tf.float32 or tf.double
a = tf.constant([[1.0,2],[3.0,4]],dtype = tf.float32)
tf.linalg.inv(a)
```

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[-2.0000002 ,  1.0000001 ],
       [ 1.5000001 , -0.50000006]], dtype=float32)>
```

```python
# Matrix trace
a = tf.constant([[1.0,2],[3,4]])
tf.linalg.trace(a)
```

```
<tf.Tensor: shape=(), dtype=float32, numpy=5.0>
```

```python
# Matrix norm
a = tf.constant([[1.0,2],[3,4]])
tf.linalg.norm(a)
```

```
<tf.Tensor: shape=(), dtype=float32, numpy=5.477226>
```

```python
# Determinant
a = tf.constant([[1.0,2],[3,4]])
tf.linalg.det(a)
```

```
<tf.Tensor: shape=(), dtype=float32, numpy=-2.0>
```

```python
# Eigenvalues
a = tf.constant([[1.0,2],[5,4]])
tf.linalg.eigvals(a)
```

```
<tf.Tensor: shape=(2,), dtype=complex64, numpy=array([-0.99999994+0.j,  5.9999995 +0.j], dtype=complex64)>
```

```python
# QR decomposition
a  = tf.constant([[1.0,2.0],[3.0,4.0]],dtype = tf.float32)
q,r = tf.linalg.qr(a)
tf.print(q)
tf.print(r)
tf.print(q@r)
```

```
[[-0.316227794 -0.948683321]
 [-0.948683321 0.316227734]]
[[-3.1622777 -4.4271884]
 [0 -0.632455349]]
[[1.00000012 1.99999976]
 [3 4]]
```

```python
# SVD decomposition
a  = tf.constant([[1.0,2.0],[3.0,4.0],[5.0,6.0]], dtype = tf.float32)
s,u,v = tf.linalg.svd(a)
tf.print(u,"\n")
tf.print(s,"\n")
tf.print(v,"\n")
tf.print(u@tf.linalg.diag(s)@tf.transpose(v))

# SVD decomposition is used for dimension reduction in PCA

```

```
[[0.229847744 -0.88346082]
 [0.524744868 -0.240782902]
 [0.819642067 0.401896209]] 

[9.52551842 0.51429987] 

[[0.619629562 0.784894466]
 [0.784894466 -0.619629562]] 

[[1.00000119 2]
 [3.00000095 4.00000048]
 [5.00000143 6.00000095]]
```

```python

```

```python

```

### 4. Broadcasting Mechanism


The rules of broadcasting in TensorFlow is the same as numpy:

* 1. If two tensors are different in rank, expand the tensor with lower rank.
* 2. If two tensors has the same length along certain dimension, or one of the tensors has length 1 along certain dimension, then these two tensors are compatible along this dimension.
* 3. Two tensors that are compatible along all dimensions are able to broadcast.
* 4. After broadcasting, the length of each dimension equals to the larger one among two tensors.
* 5. When a tensor has length = 1 along any dimension while the length of corresponding dimension of the other tensor > 1, in the broadcast result, this only element is jusk like been duplicated along this dimension.

`tf.broadcast_to` expand the dimension of tensor explicitly.

```python
a = tf.constant([1,2,3])
b = tf.constant([[0,0,0],[1,1,1],[2,2,2]])
b + a  # Identical to b + tf.broadcast_to(a,b.shape)
```

```
<tf.Tensor: shape=(3, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [2, 3, 4],
       [3, 4, 5]], dtype=int32)>
```

```python
tf.broadcast_to(a,b.shape)
```

```
<tf.Tensor: shape=(3, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]], dtype=int32)>
```

```python
# Shape after broadcasting using static shape, requires arguments in TensorShape type
tf.broadcast_static_shape(a.shape,b.shape)
```

```
TensorShape([3, 3])
```

```python
# Shape after broadcasting using dynamic shape, requires arguments in Tensor type
c = tf.constant([1,2,3])
d = tf.constant([[1],[2],[3]])
tf.broadcast_dynamic_shape(tf.shape(c),tf.shape(d))
```

```
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 3], dtype=int32)>
```

```python
# Results of broadcasting
c+d # Identical to tf.broadcast_to(c,[3,3]) + tf.broadcast_to(d,[3,3])
```

```
<tf.Tensor: shape=(3, 3), dtype=int32, numpy=
array([[2, 3, 4],
       [3, 4, 5],
       [4, 5, 6]], dtype=int32)>
```

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)

```python

```
