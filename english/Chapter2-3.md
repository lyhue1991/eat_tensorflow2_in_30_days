# 2-3 Automatic Differentiate


The neural networks relies on back propagations to calculate gradients and update the parameters in the network. Gradient calculation is complicated which is easy to incur mistakes.

The framework of deeplearning helps us to calculate gradient automatically.

`tf.GradientTape` is usually used to record forward calculation in Tensorflow, and reverse this "tape" to obtain the gradient.

This is the automatic differentiate in TensorFlow.


### 1. Calculate the Derivative Using the Gradient Tape

```python
import tensorflow as tf
import numpy as np 

# Calculate the derivative of f(x) = a*x**2 + b*x + c

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    y = a*tf.pow(x,2) + b*x + c
    
dy_dx = tape.gradient(y,x)
print(dy_dx)
```

```
tf.Tensor(-2.0, shape=(), dtype=float32)
```

```python

```

```python
# Use watch to calculate derivatives of the constant tensor

with tf.GradientTape() as tape:
    tape.watch([a,b,c])
    y = a*tf.pow(x,2) + b*x + c
    
dy_dx,dy_da,dy_db,dy_dc = tape.gradient(y,[x,a,b,c])
print(dy_da)
print(dy_dc)

```

```
tf.Tensor(0.0, shape=(), dtype=float32)
tf.Tensor(1.0, shape=(), dtype=float32)
```

```python

```

```python
# Calculate the second order derivative
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:   
        y = a*tf.pow(x,2) + b*x + c
    dy_dx = tape1.gradient(y,x)   
dy2_dx2 = tape2.gradient(dy_dx,x)

print(dy2_dx2)
```

```
tf.Tensor(2.0, shape=(), dtype=float32)
```

```python

```

```python
# Use it in the autograph

@tf.function
def f(x):   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    
    # Convert the type of the variable to tf.float32
    x = tf.cast(x,tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a*tf.pow(x,2)+b*x+c
    dy_dx = tape.gradient(y,x) 
    
    return((dy_dx,y))

tf.print(f(tf.constant(0.0)))
tf.print(f(tf.constant(1.0)))
```

```
(-2, 1)
(0, 0)
```

```python

```

### 2. Calculate the Minimal Value Through the Gradient Tape and the Optimizer

```python
# Calculate the minimal value of f(x) = a*x**2 + b*x + c
# Use optimizer.apply_gradients

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a*tf.pow(x,2) + b*x + c
    dy_dx = tape.gradient(y,x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
    
tf.print("y =",y,"; x =",x)
```

```
y = 0 ; x = 0.999998569
```

```python

```

```python
# Calculate the minimal value off(x) = a*x**2 + b*x + c
# Use optimizer.minimize
# This optimizer.minimize is identical to calculating gradient using tape, then call apply_gradient

x = tf.Variable(0.0,name = "x",dtype = tf.float32)

#Note that f() has no argument
def f():   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2)+b*x+c
    return(y)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)   
for _ in range(1000):
    optimizer.minimize(f,[x])   
    
tf.print("y =",f(),"; x =",x)
```

```
y = 0 ; x = 0.999998569
```

```python

```

```python
# Calculate minimal value in Autograph
# Use optimizer.apply_gradients

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    
    for _ in tf.range(1000): #Note that we should use tf.range(1000) instead of range(1000) when using Autograph
        with tf.GradientTape() as tape:
            y = a*tf.pow(x,2) + b*x + c
        dy_dx = tape.gradient(y,x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
        
    y = a*tf.pow(x,2) + b*x + c
    return y

tf.print(minimizef())
tf.print(x)
```

```
0
0.999998569
```

```python

```

```python
# Calculate minimal value in Autograph
# Use optimizer.minimize

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)   

@tf.function
def f():   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2)+b*x+c
    return(y)

@tf.function
def train(epoch):  
    for _ in tf.range(epoch):  
        optimizer.minimize(f,[x])
    return(f())


tf.print(train(1000))
tf.print(x)

```

```
0
0.999998569
```

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)
