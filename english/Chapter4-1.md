# 4-1 Structural Operations of the Tensor

Tensor operation includes structural operation and mathematical operation.

The structural operation includes tensor creation, index slicing, dimension transform, combining & splitting, etc.

The mathematical operation includes scalar operation, vector operation, and matrix operation. We will also introduce the broadcasting mechanism of tensor operation.

This section is about the structural operation of tensor.


### 1. Creating Tensor


Tensor creation is similar to array creation in numpy.

```python
import tensorflow as tf
import numpy as np 
```

```python
a = tf.constant([1,2,3],dtype = tf.float32)
tf.print(a)
```

```
[1 2 3]
```
```python
b = tf.range(1,10,delta = 2)
tf.print(b)
```

```
[1 3 5 7 9]
```

```python
c = tf.linspace(0.0,2*3.14,100)
tf.print(c)
```

```
[0 0.0634343475 0.126868695 ... 6.15313148 6.21656609 6.28]
```

```python
d = tf.zeros([3,3])
tf.print(d)
```

```
[[0 0 0]
 [0 0 0]
 [0 0 0]]
```

```python
a = tf.ones([3,3])
b = tf.zeros_like(a,dtype= tf.float32)
tf.print(a)
tf.print(b)
```

```
[[1 1 1]
 [1 1 1]
 [1 1 1]]
[[0 0 0]
 [0 0 0]
 [0 0 0]]
```

```python
b = tf.fill([3,2],5)
tf.print(b)
```

```
[[5 5]
 [5 5]
 [5 5]]
```

```python
# Random numbers with uniform distribution
tf.random.set_seed(1.0)
a = tf.random.uniform([5],minval=0,maxval=10)
tf.print(a)
```

```
[1.65130854 9.01481247 6.30974197 4.34546089 2.9193902]
```

```python
# Random numbers with normal distribution
b = tf.random.normal([3,3],mean=0.0,stddev=1.0)
tf.print(b)
```

```
[[0.403087884 -1.0880208 -0.0630953535]
 [1.33655667 0.711760104 -0.489286453]
 [-0.764221311 -1.03724861 -1.25193381]]
```

```python
# Random numbers with normal distribution and truncate within the range 2X standard deviation
c = tf.random.truncated_normal((5,5), mean=0.0, stddev=1.0, dtype=tf.float32)
tf.print(c)
```

```
[[-0.457012236 -0.406867266 0.728577733 -0.892977774 -0.369404584]
 [0.323488563 1.19383323 0.888299048 1.25985599 -1.95951891]
 [-0.202244401 0.294496894 -0.468728036 1.29494202 1.48142183]
 [0.0810953453 1.63843894 0.556645 0.977199793 -1.17777884]
 [1.67368948 0.0647980496 -0.705142677 -0.281972528 0.126546144]]
```

```python
# Special matrix
I = tf.eye(3,3) # Identity matrix
tf.print(I)
tf.print(" ")
t = tf.linalg.diag([1,2,3]) # Diagonal matrix
tf.print(t)
```

```
[[1 0 0]
 [0 1 0]
 [0 0 1]]
 
[[1 0 0]
 [0 2 0]
 [0 0 3]]
```

```python

```

### 2. Indexing and Slicing


The indexing and slicing of tensor is the same as numpy, and slicing supports default parameters and ellipsis.

Data type of `tf.Variable` supports indexing and slicing to modify values of certain elements.

For referencing a continuous portion of a tensor, `tf.slice` is recommended.

On the other hand, for the irregular slicing shape, `tf.gather`, `tf.gather_nd`, `tf.boolean_mask` are recommended.

The method `tf.boolean_mask` is powerful, it functions as both `tf.gather` and `tf.gather_nd`, and supports boolean indexing.

For the purpose of creating a new tensor through modifying certain elements in an existing tensor, `tf.where` and `tf.scatter_nd` can be used.

```python
tf.random.set_seed(3)
t = tf.random.uniform([5,5],minval=0,maxval=10,dtype=tf.int32)
tf.print(t)
```

```
[[4 7 4 2 9]
 [9 1 2 4 7]
 [7 2 7 4 0]
 [9 6 9 7 2]
 [3 7 0 0 3]]
```

```python
# Row 0
tf.print(t[0])
```

```
[4 7 4 2 9]
```

```python
# Last row
tf.print(t[-1])
```

```
[3 7 0 0 3]
```

```python
# Row 1 Column 3
tf.print(t[1,3])
tf.print(t[1][3])
```

```
4
4
```

```python
# From row 1 to row 3
tf.print(t[1:4,:])
tf.print(tf.slice(t,[1,0],[3,5])) #tf.slice(input,begin_vector,size_vector)
```

```
[[9 1 2 4 7]
 [7 2 7 4 0]
 [9 6 9 7 2]]
[[9 1 2 4 7]
 [7 2 7 4 0]
 [9 6 9 7 2]]
```

```python
# From row 1 to the last row, and from column 0 to the last but one with an increment of 2
tf.print(t[1:4,:4:2])
```

```
[[9 2]
 [7 7]
 [9 9]]
```

```python
# Variable supports modifying elements through indexing and slicing
x = tf.Variable([[1,2],[3,4]],dtype = tf.float32)
x[1,:].assign(tf.constant([0.0,0.0]))
tf.print(x)
```

```
[[1 2]
 [0 0]]
```

```python
a = tf.random.uniform([3,3,3],minval=0,maxval=10,dtype=tf.int32)
tf.print(a)
```

```
[[[7 3 9]
  [9 0 7]
  [9 6 7]]

 [[1 3 3]
  [0 8 1]
  [3 1 0]]

 [[4 0 6]
  [6 2 2]
  [7 9 5]]]
```

```python
# Ellipsis represents multiple colons
tf.print(a[...,1])
# This is equal to
tf.print(a[:,:,1])
```

```
[[3 0 6]
 [3 8 1]
 [0 2 9]]
[[3 0 6]
 [3 8 1]
 [0 2 9]]
```


The examples above are regular slicing; for irregular slicing, `tf.gather`, `tf.gather_nd`, `tf.boolean_mask` can be used.

Here is an example of student's grade records. There are 4 classes, 10 students in each class, and 7 courses for each student, which could be represented as a tensor with a dimension of 4×10×7.

```python
scores = tf.random.uniform((4,10,7),minval=0,maxval=100,dtype=tf.int32)
tf.print(scores)
```

```
[[[52 82 66 ... 17 86 14]
  [8 36 94 ... 13 78 41]
  [77 53 51 ... 22 91 56]
  ...
  [11 19 26 ... 89 86 68]
  [60 72 0 ... 11 26 15]
  [24 99 38 ... 97 44 74]]

 [[79 73 73 ... 35 3 81]
  [83 36 31 ... 75 38 85]
  [54 26 67 ... 60 68 98]
  ...
  [20 5 18 ... 32 45 3]
  [72 52 81 ... 88 41 20]
  [0 21 89 ... 53 10 90]]

 [[52 80 22 ... 29 25 60]
  [78 71 54 ... 43 98 81]
  [21 66 53 ... 97 75 77]
  ...
  [6 74 3 ... 53 65 43]
  [98 36 72 ... 33 36 81]
  [61 78 70 ... 7 59 21]]

 [[56 57 45 ... 23 15 3]
  [35 8 82 ... 11 59 97]
  [44 6 99 ... 81 60 27]
  ...
  [76 26 35 ... 51 8 17]
  [33 52 53 ... 78 37 31]
  [71 27 44 ... 0 52 16]]]
```

```python
# Extract all the grades of the 0th, 5th and 9th students in each class.
p = tf.gather(scores,[0,5,9],axis=1)
tf.print(p)
```

```
[[[52 82 66 ... 17 86 14]
  [24 80 70 ... 72 63 96]
  [24 99 38 ... 97 44 74]]

 [[79 73 73 ... 35 3 81]
  [46 10 94 ... 23 18 92]
  [0 21 89 ... 53 10 90]]

 [[52 80 22 ... 29 25 60]
  [19 12 23 ... 87 86 25]
  [61 78 70 ... 7 59 21]]

 [[56 57 45 ... 23 15 3]
  [6 41 79 ... 97 43 13]
  [71 27 44 ... 0 52 16]]]
```

```python
# Extract the grades of the 1st, 3rd and 6th courses of the 0th, 5th and 9th students in each class.
q = tf.gather(tf.gather(scores,[0,5,9],axis=1),[1,3,6],axis=2)
tf.print(q)
```

```
[[[82 55 14]
  [80 46 96]
  [99 58 74]]

 [[73 48 81]
  [10 38 92]
  [21 86 90]]

 [[80 57 60]
  [12 34 25]
  [78 71 21]]

 [[57 75 3]
  [41 47 13]
  [27 96 16]]]
```

```python
# Extract all the grades of the 0th student in the 0th class, the 4th student in the 2nd class, and the 6th student in the 3rd class.
# Then length of the parameter indices equals to the number of samples, and the each element of indices is the coordinate of each sample.
s = tf.gather_nd(scores,indices = [(0,0),(2,4),(3,6)])
s
```

```
<tf.Tensor: shape=(3, 7), dtype=int32, numpy=
array([[52, 82, 66, 55, 17, 86, 14],
       [99, 94, 46, 70,  1, 63, 41],
       [46, 83, 70, 80, 90, 85, 17]], dtype=int32)>
```


The function of `tf.gather` and `tf.gather_nd` as shown above could be achieved through `tf.boolean_mask`.

```python
# Extract all the grades of the 0th, 5th and 9th students in each class.
p = tf.boolean_mask(scores,[True,False,False,False,False,
                            True,False,False,False,True],axis=1)
tf.print(p)
```

```
[[[52 82 66 ... 17 86 14]
  [24 80 70 ... 72 63 96]
  [24 99 38 ... 97 44 74]]

 [[79 73 73 ... 35 3 81]
  [46 10 94 ... 23 18 92]
  [0 21 89 ... 53 10 90]]

 [[52 80 22 ... 29 25 60]
  [19 12 23 ... 87 86 25]
  [61 78 70 ... 7 59 21]]

 [[56 57 45 ... 23 15 3]
  [6 41 79 ... 97 43 13]
  [71 27 44 ... 0 52 16]]]
```

```python
# Extract all the grades of the 0th student in the 0th class, the 4th student in the 2nd class, and the 6th student in the 3rd class.
s = tf.boolean_mask(scores,
    [[True,False,False,False,False,False,False,False,False,False],
     [False,False,False,False,False,False,False,False,False,False],
     [False,False,False,False,True,False,False,False,False,False],
     [False,False,False,False,False,False,True,False,False,False]])
tf.print(s)
```

```
[[52 82 66 ... 17 86 14]
 [99 94 46 ... 1 63 41]
 [46 83 70 ... 90 85 17]]
```

```python
# Boolean indexing using tf.boolean_mask

# Find all elements that are less than 0 in the matrix
c = tf.constant([[-1,1,-1],[2,2,-2],[3,-3,3]],dtype=tf.float32)
tf.print(c,"\n")

tf.print(tf.boolean_mask(c,c<0),"\n") 
tf.print(c[c<0]) # This is the syntactic sugar of boolean_mask for boolean indexing.
```

```
[[-1 1 -1]
 [2 2 -2]
 [3 -3 3]] 

[-1 -1 -2 -3] 

[-1 -1 -2 -3]
```

```python

```

The methods shown above are able to extract part of the elements in the tensor, but are not able to create new tensors through modification of these elements.

The method `tf.where` and `tf.scatter_nd` should be used for this purpose.

`tf.where` is the tensor version of `if`; on the other hand, this method is able to find the coordinate of all the elements that statisfy certain conditions.

`tf.scatter_nd` works in an opposite way to the method `tf.gather_nd`. The latter collects the elements according to the given coordinate, while the former inserts values on the given positions in an all-zero tensor with a known shape.

```python
# Find elements that are less than 0, create a new tensor by replacing these elements with np.nan.
# tf.where is similar to np.where, which is the "if" for the tensors

c = tf.constant([[-1,1,-1],[2,2,-2],[3,-3,3]],dtype=tf.float32)
d = tf.where(c<0,tf.fill(c.shape,np.nan),c) 
d
```

```
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[nan,  1., nan],
       [ 2.,  2., nan],
       [ 3., nan,  3.]], dtype=float32)>
```

```python

```

```python
# The method where returns all the coordinates that satisfy the condition if there is only one argument
indices = tf.where(c<0)
indices
```

```
<tf.Tensor: shape=(4, 2), dtype=int64, numpy=
array([[0, 0],
       [0, 2],
       [1, 2],
       [2, 1]])>
```

```python
# Create a new tensor by replacing the value of two tensor elements located at [0,0] [2,1] as 0.
d = c - tf.scatter_nd([[0,0],[2,1]],[c[0,0],c[2,1]],c.shape)
d
```

```
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[ 0.,  1., -1.],
       [ 2.,  2., -2.],
       [ 3.,  0.,  3.]], dtype=float32)>

```

```python
# The method scatter_nd functions inversly to gather_nd
# This method can be used to insert values on the given positions in an all-zero tensor with a known shape.
indices = tf.where(c<0)
tf.scatter_nd(indices,tf.gather_nd(c,indices),c.shape)
```

```
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[-1.,  0., -1.],
       [ 0.,  0., -2.],
       [ 0., -3.,  0.]], dtype=float32)>
```

```python

```

### 3. Dimension Transform


The functions that are related to dimension transform include `tf.reshape`, `tf.squeeze`, `tf.expand_dims`, `tf.transpose`.

`tf.reshape` is used to alter the shape of the tensor.

`tf.squeeze` is used to reduce the number of dimensions.

`tf.expand_dims` is used to increase the number of dimensions.

`tf.transpose` is used to exchange the order of the dimensions.



tf.reshape changes the shape of the tensor, but will not change the order of elements stored in the memory, thus this operation is extremely fast and reversible.

```python
a = tf.random.uniform(shape=[1,3,3,2],
                      minval=0,maxval=255,dtype=tf.int32)
tf.print(a.shape)
tf.print(a)
```

```
TensorShape([1, 3, 3, 2])
[[[[135 178]
   [26 116]
   [29 224]]

  [[179 219]
   [153 209]
   [111 215]]

  [[39 7]
   [138 129]
   [59 205]]]]
```

```python
# Reshape into (3,6)
b = tf.reshape(a,[3,6])
tf.print(b.shape)
tf.print(b)
```

```
TensorShape([3, 6])
[[135 178 26 116 29 224]
 [179 219 153 209 111 215]
 [39 7 138 129 59 205]]
```




```python
# Reshape back to (1,3,3,2)
c = tf.reshape(b,[1,3,3,2])
tf.print(c)
```

```
[[[[135 178]
   [26 116]
   [29 224]]

  [[179 219]
   [153 209]
   [111 215]]

  [[39 7]
   [138 129]
   [59 205]]]]
```

```python

```

When there is only one element on a certain dimension, `tf.squeeze` eliminates this dimension.

It won't change the order of the stored elements in the memory, which is similar to `tf.reshape`.

The elements in a tensor is stored linearly, usually the adjacent elements in the same dimension use adjacent physical addresses.

```python
s = tf.squeeze(a)
tf.print(s.shape)
tf.print(s)
```

```
TensorShape([3, 3, 2])
[[[135 178]
  [26 116]
  [29 224]]

 [[179 219]
  [153 209]
  [111 215]]

 [[39 7]
  [138 129]
  [59 205]]]
```

```python
d = tf.expand_dims(s,axis=0) # Insert an extra dimension to the 0th dim with length = 1
d
```

```
<tf.Tensor: shape=(1, 3, 3, 2), dtype=int32, numpy=
array([[[[135, 178],
         [ 26, 116],
         [ 29, 224]],

        [[179, 219],
         [153, 209],
         [111, 215]],

        [[ 39,   7],
         [138, 129],
         [ 59, 205]]]], dtype=int32)>
```


`tf.transpose` swaps the dimensions in the tensor; unlike `tf.shape`, it will change the order of the elements in the memory.

`tf.transpose` is usually used for converting image format of storage.

```python
# Batch,Height,Width,Channel
a = tf.random.uniform(shape=[100,600,600,4],minval=0,maxval=255,dtype=tf.int32)
tf.print(a.shape)

# Transform to the order as Channel,Height,Width,Batch
s= tf.transpose(a,perm=[3,1,2,0])
tf.print(s.shape)
```

```
TensorShape([100, 600, 600, 4])
TensorShape([4, 600, 600, 100])

```

```python

```

### 4. Combining and Splitting


We can use `tf.concat` and `tf.stack` methods to combine multiple tensors, and use `tf.split` to split a tensor into multiple ones, which are similar as those in numpy.

`tf.concat` is slightly different to `tf.stack`: `tf.concat` is concatination and does not increase the number of dimensions, while `tf.stack` is stacking and increases the number of dimensions.

```python
a = tf.constant([[1.0,2.0],[3.0,4.0]])
b = tf.constant([[5.0,6.0],[7.0,8.0]])
c = tf.constant([[9.0,10.0],[11.0,12.0]])

tf.concat([a,b,c],axis = 0)
```

```
<tf.Tensor: shape=(6, 2), dtype=float32, numpy=
array([[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.],
       [ 7.,  8.],
       [ 9., 10.],
       [11., 12.]], dtype=float32)>
```

```python
tf.concat([a,b,c],axis = 1)
```

```
<tf.Tensor: shape=(2, 6), dtype=float32, numpy=
array([[ 1.,  2.,  5.,  6.,  9., 10.],
       [ 3.,  4.,  7.,  8., 11., 12.]], dtype=float32)>
```

```python
tf.stack([a,b,c])
```

```
<tf.Tensor: shape=(3, 2, 2), dtype=float32, numpy=
array([[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]],

       [[ 9., 10.],
        [11., 12.]]], dtype=float32)>
```

```python
tf.stack([a,b,c],axis=1)
```

```
<tf.Tensor: shape=(2, 3, 2), dtype=float32, numpy=
array([[[ 1.,  2.],
        [ 5.,  6.],
        [ 9., 10.]],

       [[ 3.,  4.],
        [ 7.,  8.],
        [11., 12.]]], dtype=float32)>
```

```python
a = tf.constant([[1.0,2.0],[3.0,4.0]])
b = tf.constant([[5.0,6.0],[7.0,8.0]])
c = tf.constant([[9.0,10.0],[11.0,12.0]])

c = tf.concat([a,b,c],axis = 0)
```

`tf.split` is the inverse of `tf.concat`. It allows even splitting with given number of portions, or uneven splitting with given size of each portion.

```python
#tf.split(value,num_or_size_splits,axis)
tf.split(c,3,axis = 0)  # Even splitting with given number of portions
```

```
[<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[1., 2.],
        [3., 4.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[5., 6.],
        [7., 8.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[ 9., 10.],
        [11., 12.]], dtype=float32)>]
```

```python
tf.split(c,[2,2,2],axis = 0) # Splitting with given size of each portion.
```

```
[<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[1., 2.],
        [3., 4.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[5., 6.],
        [7., 8.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[ 9., 10.],
        [11., 12.]], dtype=float32)>]
```

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)
