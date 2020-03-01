# 三、TensorFlow的层次结构


本章我们介绍TensorFlow中5个不同的层次结构：即硬件层，内核层，低阶API，中阶API，高阶API。并以线性回归为例，直观对比展示在不同层级实现模型的特点。

TensorFlow的层次结构从低到高可以分成如下五层。

最底层为硬件层，TensorFlow支持CPU、GPU或TPU加入计算资源池。

第二层为C++实现的内核，kernel可以跨平台分布运行。

第三层为Python实现的操作符，提供了封装C++内核的低级API指令，主要包括各种张量操作算子、计算图、自动微分.
如tf.Variable,tf.constant,tf.function,tf.GradientTape,tf.nn.softmax...
如果把模型比作一个房子，那么第三层API就是【模型之砖】。

第四层为Python实现的模型组件，对低级API进行了函数封装，主要包括各种模型层，损失函数，优化器，数据管道，特征列等等。
如tf.keras.layers,tf.keras.losses,tf.keras.metrics,tf.keras.optimizers,tf.data.DataSet,tf.feature_column...
如果把模型比作一个房子，那么第四层API就是【模型之墙】。

第五层为Python实现的模型成品，一般为按照OOP方式封装的高级API，主要为tf.keras.models提供的模型的类接口。
如果把模型比作一个房子，那么第五层API就是模型本身，即【模型之屋】。


<img src="./data/tensorflow_structure.jpg">


如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"Python与算法之美"下留言。作者时间和精力有限，会酌情予以回复。

![](./data/Python与算法之美logo.jpg)
