# 5-5 losses

In general, the target function in supervised learning consists of loss function and regularization term. (Target = Loss + Regularization)

For the keras model, the regularization term of the target function is usually designated in each layer, such as using `kernel_regularizer` and `bias_regularizer` parameters in `Dense` layer to sepecify using l1 or l2 norm. On the other hand, `kernel_constraint` and `bias_constraint` parameters can limit the range of the weights, which is also a method of regularization.

Loss function is designated during the compilation of the model. For the regression models, the most popular loss function is the mean squared error `mean_squared_error`.

For binary classification model, the most popular loss function is binary cross entropy `binary_crossentropy`.

For multiple classification model, when the labels are one-hot encoded, we should use categorical cross entropy `categorical_crossentropy` as loss function; for the category with ordinal encoding, the sparse categorical cross entropy `sparse_categorical_crossentropy` should be used as the loss funcion.

We may define customized loss function when necessary. The customzed loss function requires two tensors `y_true` and `y_pred` as input,and it output a scalar as the value of the caluclated loss function.


```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models,losses,regularizers,constraints
```

### 1. Loss Function and Regularization Term

```python
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01), 
                activity_regularizer=regularizers.l1(0.01),
                kernel_constraint = constraints.MaxNorm(max_value=2, axis=0))) 
model.add(layers.Dense(10,
        kernel_regularizer=regularizers.l1_l2(0.01,0.01),activation = "sigmoid"))
model.compile(optimizer = "rmsprop",
        loss = "binary_crossentropy",metrics = ["AUC"])
model.summary()

```

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                4160      
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 4,810
Trainable params: 4,810
Non-trainable params: 0
_________________________________________________________________
```


### 2. Pre-defined Loss Function


There are two types of implementation of the pre-defined loss function: class-type or function-type.

e.g. `CategoricalCrossentropy` and `categorical_crossentropy` are both categorical cross entropy; the former is the implementation by class, and the latter is the by function.

The most frequently used pre-defined loss functions are:

* mean_squared_error (Mean Squared Error, for regression, dubbed as "mse", class-type and function-type implementations are `MeanSquaredError` and `MSE`, respectively)

* mean_absolute_error (Mean Absolute Error, for regression, dubbed as "mae", class-type and function-type implementations are `MeanAbsoluteError` and `MAE`, respectively)

* mean_absolute_percentage_error (Mean Absolute Percentage Error, for regression dubed as "mape", class-type and function-type implementations are `MeanAbsolutePercentageError` and `MAPE`, respectively)

* Huber (Huber Loss，for regression, performance is between "mse" and "mae", robust to outliers, thus has advantages comparint to "mse"; implemented only in class)

* binary_crossentropy (Binary Cross Entropy, for binary classification; the class-type implementation is `BinaryCrossentropy`)

* categorical_crossentropy (Categorical Cross Entropy, for multiple classification, requires one-hot encoding for the label; the class-type implementation is `CategoricalCrossentropy`)

* sparse_categorical_crossentropy (Sparse Categorical Cross Entropy, used for multiple classification, requires ordinal encoding; the class-type implementation is `SparseCategoricalCrossentropy`)

* hinge (Hinge loss function, for binary classification, famous for the application as loss function in Support Vector Machine (SVM); the class-type implementation is `Hinge`)

* kld (Kullback-Leibler divergence loss, usually used as the loss function in the expectation maximization (EM) algorithm; it is a measurement of the difference between two probability distributions. The class-type and function-type implementations are `KLDivergence` and `KLD`, respectively)

* cosine_similarity (Cosine similarity, for multiple classification; the class-type implementation is `CosineSimilarity`)

```python

```

### 3. Customized Loss Function


The customzed loss function requires two tensors `y_true` and `y_pred` as input,and it output a scalar as the value of the caluclated loss function.

It is also possible to customize loss function through inheriting from the base class `tf.keras.losses.Loss` and rewrite the `call` method to implement the calculation of loss.

Here is an example of customized implementation to the Focal Loss, which is an improvement of `binary_crossentropy` loss function.

Focal Loss results better comparing to the binary cross entropy, given the condition of unbalanced category and many easy samples in training data.

It has two adjustable parameters，alpha and gamma. The aim of alpha is to decay the weight of negative samples，and gamma to decay the weight of the easy samples. 

So the model will then focal its weight on the positive samples and hard samples. This is why the loss is called focal loss. 

You may refer to the following article for details of this topic: [Understand Focal Loss and GHM in 5 minutes](https://www.zhihu.com/question/63581984)

```python
def focal_loss(gamma=2., alpha=0.75):
    
    def focal_loss_fixed(y_true, y_pred):
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        loss = tf.reduce_sum(alpha_factor * modulating_factor * bce,axis = -1 )
        return loss
    return focal_loss_fixed

```

```python
class FocalLoss(losses.Loss):
    
    def __init__(self,gamma=2.0,alpha=0.75,name = "focal_loss"):
        self.gamma = gamma
        self.alpha = alpha

    def call(self,y_true,y_pred):
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        loss = tf.reduce_sum(alpha_factor * modulating_factor * bce,axis = -1 )
        return loss
```

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)

```python

```
