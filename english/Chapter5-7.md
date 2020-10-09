# 5-7 optimizers

There is a group of magic cooks in machine learning. Their daily life looks like:

They grab some raw material (data), put them into a pot (model), light some fire (optimization algorithm), and wait until the cuisine is ready.

However, anyone who has cooking experience knows that fire controlling is the key part. Even using same material with the same recipe, different fire level leads to totally different results: medium well, burnt, or still raw.

This theroy on cooking also applies to the machine learning. The choice of the optimization algorithm determines the final performance of the final model. An unsatisfying performance is not necessarily due to the problem of feature or model designing, instead, it might be attributed to the choice of optimization algorithm.

The evolution of the optimization algorithm for the deep learning is: SGD -> SGDM -> NAG ->Adagrad -> Adadelta(RMSprop) -> Adam -> Nadam

You may refer to the following article to for more details ["Understand the differences in optimization algorthms with just one framework: SGD/AdaGrad/Adam"](https://zhuanlan.zhihu.com/p/32230623)

For the beginners, choosing Adam as the optimizer and using the default parameters will set everything for you.

Some researchers who are chaising better metrics for publications could use Adam as the initial optimizer and use SGD later for fine-tuning the parameters for better performance.

There are some cutting-edge optimization algorithms claiming a better performance, e.g. LazyAdam, Look-ahead, RAdam, Ranger, etc.


```python

```

### 1. How To Use the Optimizer


Optimizer accepts variables and corresponding gradient through `apply_gradients` method to iterate over the given variables. Another way is using `minimize` method to optimize the target function iteratively.

Another common way is passing the optimizer to the `Model` of keras, and call `model.fit` method to optimize the loss function.

A variable named `optimizer.iterations` will be created during optimizer initialization to record the number of iteration. Thus the optimizer should be created outside the decorator `@tf.function` with the same reason as `tf.Variable`.

```python
import tensorflow as tf
import numpy as np 

# Time stamp
@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts%(24*60*60)

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
    tf.print("=========="*8,end = "")
    tf.print(timestring)
    
```

```python
# The minimal value of f(x) = a*x**2 + b*x + c

# Here we use optimizer.apply_gradients

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    
    while tf.constant(True): 
        with tf.GradientTape() as tape:
            y = a*tf.pow(x,2) + b*x + c
        dy_dx = tape.gradient(y,x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
        
        # Condition of terminating the iteration
        if tf.abs(dy_dx)<tf.constant(0.00001):
            break
            
        if tf.math.mod(optimizer.iterations,100)==0:
            printbar()
            tf.print("step = ",optimizer.iterations)
            tf.print("x = ", x)
            tf.print("")
                
    y = a*tf.pow(x,2) + b*x + c
    return y

tf.print("y =",minimizef())
tf.print("x =",x)
```

```python

```

```python
# Minimal value of f(x) = a*x**2 + b*x + c

# Here we use optimizer.minimize

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)   

def f():   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2)+b*x+c
    return(y)

@tf.function
def train(epoch = 1000):  
    for _ in tf.range(epoch):  
        optimizer.minimize(f,[x])
    tf.print("epoch = ",optimizer.iterations)
    return(f())

train(1000)
tf.print("y = ",f())
tf.print("x = ",x)

```

```python

```

```python
# Minimal value of f(x) = a*x**2 + b*x + c
# Here we use model.fit

tf.keras.backend.clear_session()

class FakeModel(tf.keras.models.Model):
    def __init__(self,a,b,c):
        super(FakeModel,self).__init__()
        self.a = a
        self.b = b
        self.c = c
    
    def build(self):
        self.x = tf.Variable(0.0,name = "x")
        self.built = True
    
    def call(self,features):
        loss  = self.a*(self.x)**2+self.b*(self.x)+self.c
        return(tf.ones_like(features)*loss)
    
def myloss(y_true,y_pred):
    return tf.reduce_mean(y_pred)

model = FakeModel(tf.constant(1.0),tf.constant(-2.0),tf.constant(1.0))

model.build()
model.summary()

model.compile(optimizer = 
              tf.keras.optimizers.SGD(learning_rate=0.01),loss = myloss)
history = model.fit(tf.zeros((100,2)),
                    tf.ones(100),batch_size = 1,epochs = 10)  # Iterate for 1000 times

```

```python
tf.print("x=",model.x)
tf.print("loss=",model(tf.constant(0.0)))
```

```python

```

### 2. Pre-defined Optimizers


The evolution of the optimization algorithm for the deep learning is: SGD -> SGDM -> NAG ->Adagrad -> Adadelta(RMSprop) -> Adam -> Nadam

There are corresponding classes in `keras.optimizers` sub-module as the implementations of these optimizers.

* `SGD`, the default parameters is for a pure SGD. For a non-zero parameter `momentum`, the optimizer changes to SGDM since it considers the first-order momentum. For `nesterov` = True, the optimizer changes to NAG (Nesterov Accelerated Gradient), which calculates the gradient of the one further step.

* `Adagrad`, considers the second-order momentum and equipted with self-adaptive learning rate; the drawback is a slow learning rate at a later stage or early ceasing of learning due to the monotonically desending leanring rate.

* `RMSprop`, considers the second-order momentum and equipted with self-adaptive learning rate; improves the `Adagrad` through exponential smoothing, which only cnosiders the second-order momentum in a given window length.

* `Adadelta`, considers the second-order momentum, similar as `RMSprop` but more complicated with an improved self-adaption.

* `Adam`, consider both the first-order and the second-order momentum; it improves `RMSprop` by including first-order momentum.

* `Nadam`, improves `Adam` by including Nesterov Acceleration.

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)

```python

```

```python

```
