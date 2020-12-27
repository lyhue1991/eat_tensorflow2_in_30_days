# 6-6,使用tensorflow-serving部署模型

TensorFlow训练好的模型以tensorflow原生方式保存成protobuf文件后可以用许多方式部署运行。

例如：通过 tensorflow-js 可以用javascrip脚本加载模型并在浏览器中运行模型。

通过 tensorflow-lite 可以在移动和嵌入式设备上加载并运行TensorFlow模型。

通过 tensorflow-serving 可以加载模型后提供网络接口API服务，通过任意编程语言发送网络请求都可以获取模型预测结果。

通过 tensorFlow for Java接口，可以在Java或者spark(scala)中调用tensorflow模型进行预测。

我们主要介绍tensorflow serving部署模型、使用spark(scala)调用tensorflow模型的方法。

```python

```

### 〇，tensorflow serving模型部署概述

<!-- #region -->
使用 tensorflow serving 部署模型要完成以下步骤。

* (1) 准备protobuf模型文件。

* (2) 安装tensorflow serving。

* (3) 启动tensorflow serving 服务。

* (4) 向API服务发送请求，获取预测结果。


可通过以下colab链接测试效果《tf_serving》：
https://colab.research.google.com/drive/1vS5LAYJTEn-H0GDb1irzIuyRB8E3eWc8

<!-- #endregion -->

```python
%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import * 

```

### 一，准备protobuf模型文件

我们使用tf.keras 训练一个简单的线性回归模型，并保存成protobuf文件。

```python
import tensorflow as tf
from tensorflow.keras import models,layers,optimizers

## 样本数量
n = 800

## 生成测试用数据集
X = tf.random.uniform([n,2],minval=-10,maxval=10) 
w0 = tf.constant([[2.0],[-1.0]])
b0 = tf.constant(3.0)

Y = X@w0 + b0 + tf.random.normal([n,1],
    mean = 0.0,stddev= 2.0) # @表示矩阵乘法,增加正态扰动

## 建立模型
tf.keras.backend.clear_session()
inputs = layers.Input(shape = (2,),name ="inputs") #设置输入名字为inputs
outputs = layers.Dense(1, name = "outputs")(inputs) #设置输出名字为outputs
linear = models.Model(inputs = inputs,outputs = outputs)
linear.summary()

## 使用fit方法进行训练
linear.compile(optimizer="rmsprop",loss="mse",metrics=["mae"])
linear.fit(X,Y,batch_size = 8,epochs = 100)  

tf.print("w = ",linear.layers[1].kernel)
tf.print("b = ",linear.layers[1].bias)

## 将模型保存成pb格式文件
export_path = "./data/linear_model/"
version = "1"       #后续可以通过版本号进行模型版本迭代与管理
linear.save(export_path+version, save_format="tf") 
```

```python
#查看保存的模型文件
!ls {export_path+version}
```

```
assets	saved_model.pb	variables
```

```python
# 查看模型文件相关信息
!saved_model_cli show --dir {export_path+str(version)} --all
```

```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 2)
        name: serving_default_inputs:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['outputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict
WARNING:tensorflow:From /tensorflow-2.1.0/python3.6/tensorflow_core/python/ops/resource_variable_ops.py:1786: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None

  Function Name: '_default_save_signature'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')

  Function Name: 'call_and_return_all_conditional_losses'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
```

```python

```

### 二，安装 tensorflow serving


安装 tensorflow serving 有2种主要方法：通过Docker镜像安装，通过apt安装。

通过Docker镜像安装是最简单，最直接的方法，推荐采用。

Docker可以理解成一种容器，其上面可以给各种不同的程序提供独立的运行环境。

一般业务中用到tensorflow的企业都会有运维同学通过Docker 搭建 tensorflow serving.

无需算法工程师同学动手安装，以下安装过程仅供参考。

不同操作系统机器上安装Docker的方法可以参照以下链接。

Windows: https://www.runoob.com/docker/windows-docker-install.html

MacOs: https://www.runoob.com/docker/macos-docker-install.html

CentOS: https://www.runoob.com/docker/centos-docker-install.html

安装Docker成功后，使用如下命令加载 tensorflow/serving 镜像到Docker中

docker pull tensorflow/serving


```python

```

### 三，启动 tensorflow serving 服务

```python
!docker run -t --rm -p 8501:8501 \
    -v "/Users/.../data/linear_model/" \
    -e MODEL_NAME=linear_model \
    tensorflow/serving & >server.log 2>&1
```

```python

```

### 四，向API服务发送请求


可以使用任何编程语言的http功能发送请求，下面示范linux的 curl 命令发送请求，以及Python的requests库发送请求。

```python
!curl -d '{"instances": [[1.0, 2.0], [5.0,7.0]]}' \
    -X POST http://localhost:8501/v1/models/linear_model:predict
```

```
{
    "predictions": [[3.06546211], [6.02843142]
    ]
}
```

```python
import json,requests

data = json.dumps({"signature_name": "serving_default", "instances": [[1.0, 2.0], [5.0,7.0]]})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/linear_model:predict', 
        data=data, headers=headers)
predictions = json.loads(json_response.text)["predictions"]
print(predictions)
```

```
[[3.06546211], [6.02843142]]
```

```python

```

如果对本书内容理解上有需要进一步和作者交流的地方，欢迎在公众号"算法美食屋"下留言。作者时间和精力有限，会酌情予以回复。

也可以在公众号后台回复关键字：**加群**，加入读者交流群和大家讨论。

![算法美食屋二维码.jpg](./data/算法美食屋二维码.jpg)

```python

```

```python

```
