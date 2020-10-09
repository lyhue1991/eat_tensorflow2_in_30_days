# 5-8 callbacks

The callbacks in `tf.keras` is a class, usually specified as a parameter when use `model.fit`. It provides the extra operations at the starting or the ending of training, each epoch or each batch. These operations include record some log information, change learning rate, early termination of the training, etc.

Likewise, this callbacks parameter is also able to be specified for `model.evaluate` or `model.predict`, providing extra operations at the starting or the ending of the evaluation, prediction, or each batch. However this method is rarely used.

For the most cases, the pre-defined callbacks in the sub-module `keras.callbacks` are sufficient. It is also possible to define child class inheriting `keras.callbacks.Callbacks` to customize callbacks if necessary.

All the classes of callbacks are inheriting `keras.callbacks.Callbacks`, which contains two attributes: `params` and `model`. 

`params` is a dictionary, which records training parameters (e.g. verbosity, batch size, number of epochs, etc.). `model` is the reference to the current model.

What's more, there is an extra argument `logs` in the certain methods of the callbacks classes, such as `on_epoch_begin`, `on_batch_end`. This parameter provides certain information of current epoch or batch and are able to save the computing results. These `logs` variables are able to transfer among the functions with the same name in these callbacks classes.



### 1. Pre-defined Callbacks


* `BaseLogger`: it calcuates the mean metrics among all batches for each epoch. For those metrics with middle status in `staeful_metrics`, it uses the final metrics without calculating mean value for all the batches, and the final mean metrics is added to the variable `logs`. This callback is automatically applied to every Keras model and is applied first.

* `History`: a dictionary that records the metrics of each epoch calculated by `BaseLogger` and is returned by `model.fit`. This callback is automatically applied to every Keras model after `BaseLogger`.

* `EarlyStopping`: this callback terminates the training if the monitoring metrics are not significantly increased after certain number of epoches.

* `TensorBoard`: this callback saves the visualized log of the Tensorboard. It supports visualization of metrics, graphs and parameters in the model.

* `ModelCheckpoint`: this callback saves model after each epoch.

* `ReduceLROnPlateau`: this callback reduce the learning rate with certain rate if the monitoring metrics are not significantly increased after certain number of epoches.

* `TerminateOnNaN`: terminate the training if loss is NaN.

* `LearningRateScheduler`: it controls the learning rate before each epoch with given function between the learning rate `lr` and epoch.

* `CSVLogger`: save `logs` of each epoch in CSV file.

* `ProgbarLogger`: print the `logs` of each epoch into stardard I/O stream.



```python

```

### 2. Customized Callbacks


It is possible to write a simple callback through `callbacks.LambdaCallback`, or write a complicated callback through inheriting base class `callbacks.Callback`.

Don't hesitate to read the source code to know more details of the callbacks in `tf.Keras`.

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models,losses,metrics,callbacks
import tensorflow.keras.backend as K 

```

```python
# Example of the simple callback using LambdaCallback

import json
json_log = open('../data/keras_log.json', mode='wt', buffering=1)
json_logging_callback = callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps(dict(epoch = epoch,**logs)) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

```

```python
# Example of the complicated callback through base class inheritance. This is the source code of LearningRateScheduler.

class LearningRateScheduler(callbacks.Callback):
    
    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  
            lr = float(K.get_value(self.model.optimizer.lr))
            lr = self.schedule(epoch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        K.set_value(self.model.optimizer.lr, K.get_value(lr))
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                 'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

```

```python

```

Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)
