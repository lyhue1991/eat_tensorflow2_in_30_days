# How to eat TensorFlow2 in 30 days ?🔥🔥

快速切换至[中文版](https://github.com/lyhue1991/eat_tensorflow2_in_30_days/tree/master) 🎈

📚 URL to gitbook： https://lyhue1991.github.io/eat_tensorflow2_in_30_days

🚀 URL to github repo：https://github.com/lyhue1991/eat_tensorflow2_in_30_days


### 1. TensorFlow2 🍎 or Pytorch🔥

TL; DR

**For the engineers, priority goes to TensorFlow2.**

**For the students and researchers，first choice should be Pytorch.**

**The best way is to master both of them if having sufficient time.**


Reasons:

* 1. **Model implementation is the most important in the industry. Only deployment ONLY supports tensorflow models （not Pytorch） for the majority of the domestic enterprises (in China).** What's more, the industry prefers the models with higher availability; in most cases, they use well-validated modeling architectures with the minimized requirements of adjustment.


* 2. **Fast iterative development and publication is the most important for the researchers since they need to test a lot of new models. Pytorch has advantages in accessing and debugging comparing with TensorFlow2.** Pytorch is most frequently used in academy since 2019 with a large amount of the cutting-edge results.


* 3. Overall, TensorFlow2 and Pytorch is quite similar in programming nowadays, so mastering one helps learning the other. Mastering both framework provides you a lot more open-sourced models and helps you switching between them.

```python

```

### 2. Keras🍏 and tf.keras 🍎

TL; DR

**Keras will be discontinued in development after version 2.3.0, so use tf.keras.**


Keras is a high-level API for the deep learning frameworks. It help the users to define and training DL networks with a more intuitive way.

The Keras libraries installed by pip implement this high-level API for the backends in tensorflow, theano, CNTK, etc.

tf.keras is the high-level API just for Tensorflow, which is based on low-level APIs in Tensorflow.

Most but not all of the functions in tf.keras are the same for those in Keras (which is compatible to many kinds of backend). tf.keras has a tighter combination to TensorFlow comparing to Keras.

Keras will not update after version 2.3.0 after being acquired by Google; thus the users should use tr.keras from now on, instead of using Keras installed by pip.

```python

```

### 3. Readers 👼: Before you read this book📖


**It is suggested that the readers have foundamental knowledges of machine/deep learning and experience of modeling using Keras or TensorFlow 1.0.**

**For those who have zero experience of machine/deep learning, it is strongly suggested to refer to ["Deep Learning with Python"](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438/ref=sr_1_1?dchild=1&keywords=Deep+Learning+with+Python&qid=1586194568&sr=8-1) along with reading this book.**

["Deep Learning with Python"](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438/ref=sr_1_1?dchild=1&keywords=Deep+Learning+with+Python&qid=1586194568&sr=8-1) is written by François Chollet, the inventor of Keras. This book is based on Keras and has no machine learning related prerequisites to the reader.

"Deep Learning with Python" is easy to understand as it uses various example to demonstrate. **No mathematical equation is in this book since it focuses on cultivating the intuitive to the deep learning.**

The Chinese version of this book could be downloaded here：https://pan.baidu.com/s/1-4q6VjLTb3ZxcefyNCbjSA password：wtzo 


```python

```

### 4. Writing style 🍉


**This is a introduction reference book which is extremely friendly to human being. The lowest goal of the authors is to avoid giving up due to the difficulties, while "Don't let the readers think" is the highest target.**

This book is mainly based on the official documents of TensorFlow together with its functions.

HOWEVER, the authors made a thorough restructuring and made a lot optimizations on the demonstrations.

It is different from the official documents, which is disordered and contains both tutorial and guidance with lack of systematic logic, that our book redesigns the content according to the difficulties, readers' searching habits, and the architecture of TensorFlow. We now make it progressive for TensorFlow studying with a clear path, and an easy access to the corresponding examples.

In contrast to the verbose demonstrating code, the authors of this book try to minimize the length of the examples to make it easy for reading and implementation. What's more, most of the code cells can be executed instantaneously.

**Given the level of difficulty as 9 for learning Tensorflow through official documents, it would be reduced to 3 if learning through this book.**

This difference could be demonstrated as the following figure:

![](./data/30天吃掉那个TF2.0.jpg)


```python

```

### 5. How to learn with this book ⏰

**(1) Study Plan**

The authors wrote this book using the spare time, especially the two month unexpected "holiday" of COVID-19. Most readers should be able to completely master all the content within 30 days.

Time required everyday would be between 30 minutes to 2 hours.

This book could also be used as reference to the demonstrating libraries for the implementation of projects.

**Click the blue captions to enter the corresponding chapter.**


|Date |Contents                                                       | Difficulties   | Est. Time | Status|
|----:|:--------------------------------------------------------------|-----------:|----------:|-----:|
|&nbsp;|[**Chapter 1: Modeling Procedure of TensorFlow**](./Chapter1.md)    |⭐️   |   0hour   |✅    |
|Day 1 |  [1-1 Example: Modeling Procedure for Structured Data](./Chapter1-1.md)    | ⭐️⭐️⭐️ |   1hour    |✅    |
|Day 2 |[1-2 Example: Modeling Procedure for Images](./Chapter1-2.md)    | ⭐️⭐️⭐️⭐️  |   2hour    |✅    |
|Day 3 |  [1-3 Example: Modeling Procedure for Texts](./Chapter1-3.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    |✅    |
|Day 4 |  [1-4 Example: Modeling Procedure for Temporal Sequences](./Chapter1-4.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    |✅    |
|&nbsp;    |[**Chapter 2: Key Concepts of TensorFlow**](./Chapter2.md)  | ⭐️  |  0hour |✅  |
|Day 5 |  [2-1 Data Structure of Tensor](./Chapter2-1.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅    |
|Day 6 |  [2-2 Three Types of Graph](./Chapter2-2.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |✅    |
|Day 7 |  [2-3 Automatic Differentiate](./Chapter2-3.md)  | ⭐️⭐️⭐️   |   1hour    |✅    |
|&nbsp; |[**Chapter 3: Hierarchy of TensorFlow**](./Chapter3.md) |   ⭐️  |  0hour   |✅  |
|Day 8 |  [3-1 Low-level API: Demonstration](./Chapter3-1.md)   | ⭐️⭐️   |   0.5hour    |✅   |
|Day 9 |  [3-2 Mid-level API: Demonstration](./Chapter3-2.md)   | ⭐️⭐️⭐️   |   0.5hour    |✅  |
|Day 10 |  [3-3 High-level API: Demonstration](./Chapter3-3.md)  | ⭐️⭐️⭐️   |   0.5hour    |✅  |
|&nbsp; |[**Chapter 4: Low-level API in TensorFlow**](./四、TensorFlow的低阶API.md) |⭐️    | 0hour|🛠️  |
|Day 11|  [4-1 Structural Operations of the Tensor](./4-1,张量的结构操作.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |🛠️   |
|Day 12|  [4-2 Mathematical Operations of the Tensor](./4-2,张量的数学运算.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |🛠️  |
|Day 13|  [4-3 Rules of Using the AutoGraph](./4-3,AutoGraph的使用规范.md)| ⭐️⭐️⭐️   |   0.5hour    |🛠️  |
|Day 14|  [4-4 Mechanisms of the AutoGraph](./4-4,AutoGraph的机制原理.md)    | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |🛠️  |
|Day 15|  [4-5 AutoGraph and tf.Module](./4-5,AutoGraph和tf.Module.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |🛠️  |
|&nbsp; |[**Chapter 5: Mid-level API in TensorFlow**](./五、TensorFlow的中阶API.md) |  ⭐️  | 0hour|🛠️ |
|Day 16|  [5-1 Dataset](./5-1,数据管道Dataset.md)   | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |🛠️  |
|Day 17|  [5-2 feature_column](./5-2,特征列feature_column.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |🛠️  |
|Day 18|  [5-3 activation](./5-3,激活函数activation.md)    | ⭐️⭐️⭐️   |   0.5hour    |🛠️   |
|Day 19|  [5-4 layers](./5-4,模型层layers.md)  | ⭐️⭐️⭐️   |   1hour    |🛠️  |
|Day 20|  [5-5 losses](./5-5,损失函数losses.md)    | ⭐️⭐️⭐️   |   1hour    |🛠️  |
|Day 21|  [5-6 metrics](./5-6,评估指标metrics.md)    | ⭐️⭐️⭐️   |   1hour    |🛠️   |
|Day 22|  [5-7 optimizers](./5-7,优化器optimizers.md)    | ⭐️⭐️⭐️   |   0.5hour    |🛠️   |
|Day 23|  [5-8 callbacks](./5-8,回调函数callbacks.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |🛠️   |
|&nbsp; |[**Chapter 6: High-level API in TensorFlow**](./六、TensorFlow的高阶API.md)|    ⭐️ | 0hour|🛠️  |
|Day 24|  [6-1 Three Ways of Modeling](./6-1,构建模型的3种方法.md)   | ⭐️⭐️⭐️   |   1hour    |🛠️ |
|Day 25|  [6-2 Three Ways of Training](./6-2,训练模型的3种方法.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |🛠️   |
|Day 26|  [6-3 Model Training Using Single GPU](./6-3,使用单GPU训练模型.md)    | ⭐️⭐️   |   0.5hour    |🛠️   |
|Day 27|  [6-4 Model Training Using Multiple GPUs](./6-4,使用多GPU训练模型.md)    | ⭐️⭐️   |   0.5hour    |🛠️  |
|Day 28|  [6-5 Model Training Using TPU](./6-5,使用TPU训练模型.md)   | ⭐️⭐️   |   0.5hour    |🛠️  |
|Day 29| [6-6 Model Deploying Using tensorflow-serving](./6-6,使用tensorflow-serving部署模型.md) | ⭐️⭐️⭐️⭐️| 1hour |🛠️   |
|Day 30| [6-7 Call Tensorflow Model Using spark-scala](./6-7,使用spark-scala调用tensorflow模型.md) | ⭐️⭐️⭐️⭐️⭐️|2hour|🛠️  |
|&nbsp;| [Epilogue：A Story Between a Foodie and a Cuisine](./后记：一个吃货和一道菜的故事.md) | ⭐️|0hour|🛠️  |


```python

```

**(2) Software environment for studying**


All the source codes are tested in jupyter. It is suggested to clone the repository to local machine and run them in jupyter for an interactive learning experience.

The authors would suggest to install jupytext that converts markdown files into ipynb, so the readers would be able to open markdown files in jupyter directly.

```python
#For the readers in mainland China, using gitee will allow cloning with a faster speed
#!git clone https://gitee.com/Python_Ai_Road/eat_tensorflow2_in_30_days

#It is suggested to install jupytext that converts and run markdown files as ipynb.
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U jupytext
    
#It is also suggested to install the latest version of TensorFlow to test the demonstrating code in this book
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -U tensorflow
```

```python
import tensorflow as tf

#Note: all the codes are tested under TensorFlow 2.1
tf.print("tensorflow version:",tf.__version__)

a = tf.constant("hello")
b = tf.constant("tensorflow2")
c = tf.strings.join([a,b]," ")
tf.print(c)
```

```
tensorflow version: 2.1.0
hello tensorflow2
```

```python

```

### 6. Contact and support the author 🎈🎈


**If you find this book helpful and want to support the author, please give a star ⭐️ to this repo and don't forget to share it to your friends 😊** 

Please leave comments in the WeChat official account "Python与算法之美" (Beauty of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

![image.png](./data/Python与算法之美logo.jpg)

```python

```
