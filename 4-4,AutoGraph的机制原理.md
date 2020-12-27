# 4-4,AutoGraph的机制原理

有三种计算图的构建方式：静态计算图，动态计算图，以及Autograph。

TensorFlow 2.0主要使用的是动态计算图和Autograph。

动态计算图易于调试，编码效率较高，但执行效率偏低。

静态计算图执行效率很高，但较难调试。

而Autograph机制可以将动态图转换成静态计算图，兼收执行效率和编码效率之利。

当然Autograph机制能够转换的代码并不是没有任何约束的，有一些编码规范需要遵循，否则可能会转换失败或者不符合预期。

我们会介绍Autograph的编码规范和Autograph转换成静态图的原理。

并介绍使用tf.Module来更好地构建Autograph。

上篇我们介绍了Autograph的编码规范，本篇我们介绍Autograph的机制原理。



### 一，Autograph的机制原理


**当我们使用@tf.function装饰一个函数的时候，后面到底发生了什么呢？**

例如我们写下如下代码。

```python
import tensorflow as tf
import numpy as np 

@tf.function(autograph=True)
def myadd(a,b):
    for i in tf.range(3):
        tf.print(i)
    c = a+b
    print("tracing")
    return c
```

后面什么都没有发生。仅仅是在Python堆栈中记录了这样一个函数的签名。

**当我们第一次调用这个被@tf.function装饰的函数时，后面到底发生了什么？**

例如我们写下如下代码。

```python
myadd(tf.constant("hello"),tf.constant("world"))
```

```
tracing
0
1
2
```

<!-- #region -->
发生了2件事情，

第一件事情是创建计算图。

即创建一个静态计算图，跟踪执行一遍函数体中的Python代码，确定各个变量的Tensor类型，并根据执行顺序将算子添加到计算图中。
在这个过程中，如果开启了autograph=True(默认开启),会将Python控制流转换成TensorFlow图内控制流。
主要是将if语句转换成 tf.cond算子表达，将while和for循环语句转换成tf.while_loop算子表达，并在必要的时候添加
tf.control_dependencies指定执行顺序依赖关系。

相当于在 tensorflow1.0执行了类似下面的语句：

```python
g = tf.Graph()
with g.as_default():
    a = tf.placeholder(shape=[],dtype=tf.string)
    b = tf.placeholder(shape=[],dtype=tf.string)
    cond = lambda i: i<tf.constant(3)
    def body(i):
        tf.print(i)
        return(i+1)
    loop = tf.while_loop(cond,body,loop_vars=[0])
    loop
    with tf.control_dependencies(loop):
        c = tf.strings.join([a,b])
    print("tracing")
```

第二件事情是执行计算图。

相当于在 tensorflow1.0中执行了下面的语句：

```python
with tf.Session(graph=g) as sess:
    sess.run(c,feed_dict={a:tf.constant("hello"),b:tf.constant("world")})
```

因此我们先看到的是第一个步骤的结果：即Python调用标准输出流打印"tracing"语句。

然后看到第二个步骤的结果：TensorFlow调用标准输出流打印1,2,3。

<!-- #endregion -->

**当我们再次用相同的输入参数类型调用这个被@tf.function装饰的函数时，后面到底发生了什么？**

例如我们写下如下代码。

```python
myadd(tf.constant("good"),tf.constant("morning"))
```

```
0
1
2
```


只会发生一件事情，那就是上面步骤的第二步，执行计算图。

所以这一次我们没有看到打印"tracing"的结果。


**当我们再次用不同的的输入参数类型调用这个被@tf.function装饰的函数时，后面到底发生了什么？**

例如我们写下如下代码。

```python
myadd(tf.constant(1),tf.constant(2))
```

```
tracing
0
1
2
```


由于输入参数的类型已经发生变化，已经创建的计算图不能够再次使用。

需要重新做2件事情：创建新的计算图、执行计算图。

所以我们又会先看到的是第一个步骤的结果：即Python调用标准输出流打印"tracing"语句。

然后再看到第二个步骤的结果：TensorFlow调用标准输出流打印1,2,3。


**需要注意的是，如果调用被@tf.function装饰的函数时输入的参数不是Tensor类型，则每次都会重新创建计算图。**

例如我们写下如下代码。两次都会重新创建计算图。因此，一般建议调用@tf.function时应传入Tensor类型。

```python
myadd("hello","world")
myadd("good","morning")
```

```
tracing
0
1
2
tracing
0
1
2
```

```python

```

```python

```

### 二，重新理解Autograph的编码规范


了解了以上Autograph的机制原理，我们也就能够理解Autograph编码规范的3条建议了。

1，被@tf.function修饰的函数应尽量使用TensorFlow中的函数而不是Python中的其他函数。例如使用tf.print而不是print.

解释：Python中的函数仅仅会在跟踪执行函数以创建静态图的阶段使用，普通Python函数是无法嵌入到静态计算图中的，所以
在计算图构建好之后再次调用的时候，这些Python函数并没有被计算，而TensorFlow中的函数则可以嵌入到计算图中。使用普通的Python函数会导致
被@tf.function修饰前【eager执行】和被@tf.function修饰后【静态图执行】的输出不一致。

2，避免在@tf.function修饰的函数内部定义tf.Variable. 

解释：如果函数内部定义了tf.Variable,那么在【eager执行】时，这种创建tf.Variable的行为在每次函数调用时候都会发生。但是在【静态图执行】时，这种创建tf.Variable的行为只会发生在第一步跟踪Python代码逻辑创建计算图时，这会导致被@tf.function修饰前【eager执行】和被@tf.function修饰后【静态图执行】的输出不一致。实际上，TensorFlow在这种情况下一般会报错。

3，被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。

解释：静态计算图是被编译成C++代码在TensorFlow内核中执行的。Python中的列表和字典等数据结构变量是无法嵌入到计算图中，它们仅仅能够在创建计算图时被读取，在执行计算图时是无法修改Python中的列表或字典这样的数据结构变量的。


```python

```

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"算法美食屋"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![算法美食屋二维码.jpg](./data/算法美食屋二维码.jpg)

```python

```
