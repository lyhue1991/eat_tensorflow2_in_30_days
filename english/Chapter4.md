# Chapter 4: Low-level API in TensorFlow

Low-level API of TensorFlow includes tensor operation, graph, automatic differentiate, etc.

If we compare a model to a house, then these low-level APIs are the bricks.

We may use TensorFlow as the enhanced numpy through these low-level APIs.

TensorFlow provides a more complete set of methods comparing to numpy. These methods have higher executiing efficiency and could be further accelerated by GPU if necessary.

We gave an intuitive introduction to the low-level API in the previous sections, and we will emphasize the introduction on the tensor operation and Autograph.

Tensor operation can be devided into two sub-categories: the structural operation and the mathematical operation.

The structural operation includes tensor creation, indexing and slicing, dimension transformation, combining & splitting, etc.

The mathematical operation includes scalar operation, vector operation, and matrix operation. We will also introduce the broadcasting mechanism of tensor operation.

For the part of Autograph, we will cover its suggested rules, its mechanisms `Autograph` and `tf.Module`.


Please leave comments in the WeChat official account "Python与算法之美" (Elegant Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to reply **加群(join group)** in the WeChat official account to join the group chat with the other readers.

![image.png](../data/Python与算法之美logo.jpg)
