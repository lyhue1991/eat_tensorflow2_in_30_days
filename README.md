# 《30天吃掉那只 TensorFlow2.0 》开篇辞 🔥🔥


📚 gitbook电子书地址： https://lyhue1991.github.io/eat_tensorflow2_in_30_days

🚀 github项目地址：https://github.com/lyhue1991/eat_tensorflow2_in_30_days


### 一，TensorFlow2 🍎 还是 Pytorch🔥

先说结论:

**如果是工程师，应该优先选TensorFlow2.**

**如果是学生或者研究人员，应该优先选择Pytorch.**

**如果时间足够，最好TensorFlow2和Pytorch都要学习掌握。**


理由如下：

* 1，**在工业界最重要的是模型落地，目前国内的大部分互联网企业只支持TensorFlow模型的在线部署，不支持Pytorch。** 并且工业界更加注重的是模型的高可用性，许多时候使用的都是成熟的模型架构，调试需求并不大。


* 2，**研究人员最重要的是快速迭代发表文章，需要尝试一些较新的模型架构。而Pytorch在易用性上相比TensorFlow2有一些优势，更加方便调试。** 并且在2019年以来在学术界占领了大半壁江山，能够找到的相应最新研究成果更多。


* 3，TensorFlow2和Pytorch实际上整体风格已经非常相似了，学会了其中一个，学习另外一个将比较容易。两种框架都掌握的话，能够参考的开源模型案例更多，并且可以方便地在两种框架之间切换。

```python

```

### 二，Keras🍏 和 tf.keras 🍎

先说结论：

**Keras库在2.3.0版本后将不再更新，用户应该使用tf.keras。**


Keras可以看成是一种深度学习框架的高阶接口规范，它帮助用户以更简洁的形式定义和训练深度学习网络。

使用pip安装的Keras库同时在tensorflow,theano,CNTK等后端基础上进行了这种高阶接口规范的实现。

而tf.keras是在TensorFlow中以TensorFlow低阶API为基础实现的这种高阶接口，它是Tensorflow的一个子模块。

tf.keras绝大部分功能和兼容多种后端的Keras库用法完全一样，但并非全部，它和TensorFlow之间的结合更为紧密。

随着谷歌对Keras的收购，Keras库2.3.0版本后也将不再进行更新，用户应当使用tf.keras而不是使用pip安装的Keras.

```python

```

### 三，本书📖面向读者 👼


**本书假定读者有一定的机器学习和深度学习基础，使用过Keras或者Tensorflow1.0或者Pytorch搭建训练过模型。**

**对于没有任何机器学习和深度学习基础的同学，建议在学习本书时同步参考学习《Python深度学习》一书。**

《Python深度学习》这本书是Keras之父Francois Chollet所著，该书假定读者无任何机器学习知识，以Keras为工具，

使用丰富的范例示范深度学习的最佳实践，该书通俗易懂，**全书没有一个数学公式，注重培养读者的深度学习直觉。**。

该书电子版下载链接：https://pan.baidu.com/s/1-4q6VjLTb3ZxcefyNCbjSA 提取码：wtzo 


```python

```

### 四，本书写作风格 🍉


**本书是一本对人类用户极其友善的TensorFlow2.0入门工具书，不刻意恶心读者是本书的底限要求，Don't let me think是本书的最高追求。**

本书主要是在参考TensorFlow官方文档和函数doc文档基础上整理写成的。

但本书在篇章结构和范例选取上做了大量的优化。

不同于官方文档混乱的篇章结构，既有教程又有指南，缺少整体的编排逻辑。

本书按照内容难易程度、读者检索习惯和TensorFlow自身的层次结构设计内容，循序渐进，层次清晰，方便按照功能查找相应范例。

不同于官方文档冗长的范例代码，本书在范例设计上尽可能简约化和结构化，增强范例易读性和通用性，大部分代码片段在实践中可即取即用。

**如果说通过学习TensorFlow官方文档掌握TensorFlow2.0的难度大概是9的话，那么通过学习本书掌握TensorFlow2.0的难度应该大概是3.**

谨以下图对比一下TensorFlow官方教程与本教程的差异。

![](./data/30天吃掉那个TF2.0.jpg)


```python

```

### 五，本书学习方案 ⏰

**1，学习计划**

本书是作者利用工作之余和疫情放假期间大概2个月写成的，大部分读者应该在30天可以完全学会。

预计每天花费的学习时间在30分钟到2个小时之间。

当然，本书也非常适合作为TensorFlow的工具手册在工程落地时作为范例库参考。

**点击学习内容蓝色标题即可进入该章节。**


|日期 | 学习内容                                                       | 内容难度   | 预计学习时间 | 更新状态|
|----:|:--------------------------------------------------------------|-----------:|----------:|-----:|
|&nbsp;|[**一、TensorFlow的建模流程**](./一、TensorFlow的建模流程.md)    |⭐️   |   0hour   |✅    |
|day1 |  [1-1,结构化数据建模流程范例](./1-1,结构化数据建模流程范例.md)    | ⭐️⭐️⭐️ |   1hour    |✅    |
|day2 |[1-2,图片数据建模流程范例](./1-2,图片数据建模流程范例.md)    | ⭐️⭐️⭐️⭐️  |   2hour    |✅    |
|day3 |  [1-3,文本数据建模流程范例](./1-3,文本数据建模流程范例.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    |✅    |
|day4 |  [1-4,时间序列数据建模流程范例](./1-4,时间序列数据建模流程范例.md)   | ⭐️⭐️⭐️⭐️⭐️  |   2hour    |✅    |
|&nbsp;    |[**二、TensorFlow的核心概念**](./二、TensorFlow的核心概念.md)  | ⭐️  |  0hour |✅  |
|day5 |  [2-1,张量数据结构](./2-1,张量数据结构.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅    |
|day6 |  [2-2,三种计算图](./2-2,三种计算图.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |✅    |
|day7 |  [2-3,自动微分机制](./2-3,自动微分机制.md)  | ⭐️⭐️⭐️   |   1hour    |✅    |
|&nbsp; |[**三、TensorFlow的层次结构**](./三、TensorFlow的层次结构.md) |   ⭐️  |  0hour   |✅  |
|day8 |  [3-1,低阶API示范](./3-1,低阶API示范.md)   | ⭐️⭐️   |   0.5hour    |✅   |
|day9 |  [3-2,中阶API示范](./3-2,中阶API示范.md)   | ⭐️⭐️⭐️   |   0.5hour    |✅  |
|day10 |  [3-3,高阶API示范](./3-3,高阶API示范.md)  | ⭐️⭐️⭐️   |   0.5hour    |✅  |
|&nbsp; |[**四、TensorFlow的低阶API**](./四、TensorFlow的低阶API.md) |⭐️    | 0hour|✅  |
|day11|  [4-1,张量的结构操作](./4-1,张量的结构操作.md)  | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |✅   |
|day12|  [4-2,张量的数学运算](./4-2,张量的数学运算.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅  |
|day13|  [4-3,AutoGraph的使用规范](./4-3,AutoGraph的使用规范.md)| ⭐️⭐️⭐️   |   0.5hour    |✅  |
|day14|  [4-4,AutoGraph的机制原理](./4-4,AutoGraph的机制原理.md)    | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |✅  |
|day15|  [4-5,AutoGraph和tf.Module](./4-5,AutoGraph和tf.Module.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅  |
|&nbsp; |[**五、TensorFlow的中阶API**](./五、TensorFlow的中阶API.md) |  ⭐️  | 0hour|✅ |
|day16|  [5-1,数据管道Dataset](./5-1,数据管道Dataset.md)   | ⭐️⭐️⭐️⭐️⭐️   |   2hour    |✅  |
|day17|  [5-2,特征列feature_column](./5-2,特征列feature_column.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅  |
|day18|  [5-3,激活函数activation](./5-3,激活函数activation.md)    | ⭐️⭐️⭐️   |   0.5hour    |✅   |
|day19|  [5-4,模型层layers](./5-4,模型层layers.md)  | ⭐️⭐️⭐️   |   1hour    |✅  |
|day20|  [5-5,损失函数losses](./5-5,损失函数losses.md)    | ⭐️⭐️⭐️   |   1hour    |✅  |
|day21|  [5-6,评估指标metrics](./5-6,评估指标metrics.md)    | ⭐️⭐️⭐️   |   1hour    |✅   |
|day22|  [5-7,优化器optimizers](./5-7,优化器optimizers.md)    | ⭐️⭐️⭐️   |   0.5hour    |✅   |
|day23|  [5-8,回调函数callbacks](./5-8,回调函数callbacks.md)   | ⭐️⭐️⭐️⭐️   |   1hour    |✅   |
|&nbsp; |[**六、TensorFlow的高阶API**](./六、TensorFlow的高阶API.md)|    ⭐️ | 0hour|✅  |
|day24|  [6-1,构建模型的3种方法](./6-1,构建模型的3种方法.md)   | ⭐️⭐️⭐️   |   1hour    |✅ |
|day25|  [6-2,训练模型的3种方法](./6-2,训练模型的3种方法.md)  | ⭐️⭐️⭐️⭐️   |   1hour    |✅   |
|day26|  [6-3,使用单GPU训练模型](./6-3,使用单GPU训练模型.md)    | ⭐️⭐️   |   0.5hour    |✅   |
|day27|  [6-4,使用多GPU训练模型](./6-4,使用多GPU训练模型.md)    | ⭐️⭐️   |   0.5hour    |✅  |
|day28|  [6-5,使用TPU训练模型](./6-5,使用TPU训练模型.md)   | ⭐️⭐️   |   0.5hour    |✅  |
|day29| [6-6,使用tensorflow-serving部署模型](./6-6,使用tensorflow-serving部署模型.md) | ⭐️⭐️⭐️⭐️| 1hour |✅   |
|day30| [6-7,使用spark-scala调用tensorflow模型](./6-7,使用spark-scala调用tensorflow模型.md) | ⭐️⭐️⭐️⭐️⭐️|2hour|✅  |
|&nbsp;| [后记：一个吃货和一道菜的故事](./后记：一个吃货和一道菜的故事.md) | ⭐️|0hour|✅  |


```python

```

**2，学习环境**


本书全部源码在jupyter中编写测试通过，建议通过git克隆到本地，并在jupyter中交互式运行学习。

为了直接能够在jupyter中打开markdown文件，建议安装jupytext，将markdown转换成ipnb。

```python
#克隆本书源码到本地,使用码云镜像仓库国内下载速度更快
#!git clone https://gitee.com/Python_Ai_Road/eat_tensorflow2_in_30_days

#建议在jupyter notebook 上安装jupytext，以便能够将本书各章节markdown文件视作ipynb文件运行
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U jupytext
    
#建议在jupyter notebook 上安装最新版本tensorflow 测试本书中的代码
#!pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -U tensorflow
```

```python
import tensorflow as tf

#注：本书全部代码在tensorflow 2.1版本测试通过
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

### 六，鼓励和联系作者 🎈🎈


**如果本书对你有所帮助，想鼓励一下作者，记得给本项目加一颗星星star⭐️，并分享给你的朋友们喔😊!** 

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"Python与算法之美"下留言。作者时间和精力有限，会酌情予以回复。

![image.png](./data/Python与算法之美logo.jpg)

```python

```
