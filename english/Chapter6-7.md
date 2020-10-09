# 6-7 Call Tensorflow Model Using spark-scala

This section introduce how to use the trained TensorFlow model to predict in spark.

The prerequisite of this section is fundamental knowledge on spark and scala.

It is easier to use pyspark, since it only requires loading model with Python on each executor and predict separately.

For the consideration of the performance, the spark in scala version is the most popular.

The section shows how to use the trained TensorFlow model in spark through TensorFlow for Java.

It is possible to predit with the trained TensorFlow model in hundreds of thousands computers using the parallel computing feature of spark.




```python

```

### 0 Using TensorFlow model in spark-scala


The necessary steps for predicting with trained TensorFlow model in spark (scala) are:

(1) Preparing protobuf model file

(2) Create a spark (scala) project, insert jar package dependencies for TensorFlow in java.

(3) Loading TensorFlow model on the driver end of spark (scala) project and debug it successfully.

(4) Loading TensorFlow model on executor of spark (scala) project through RDD and debug it successfully.

(5) Loading TensorFlow model on executor of spark (scala) project through Data and debug it successfully.


```python

```

### 1. Preparing protobuf Model File


Here we train a simple linear regression model with `tf.keras` and save it as protobuf file.

```python

```

```python
import tensorflow as tf
from tensorflow.keras import models,layers,optimizers

## Number of samples
n = 800

## Generating testing dataset
X = tf.random.uniform([n,2],minval=-10,maxval=10) 
w0 = tf.constant([[2.0],[-1.0]])
b0 = tf.constant(3.0)

Y = X@w0 + b0 + tf.random.normal([n,1],mean = 0.0,stddev= 2.0)  # @ is matrix multiplication; adding Gaussian noise

## Modeling
tf.keras.backend.clear_session()
inputs = layers.Input(shape = (2,),name ="inputs") # Set the input name as "inputs"
outputs = layers.Dense(1, name = "outputs")(inputs) # Set the output name as "outputs"
linear = models.Model(inputs = inputs,outputs = outputs)
linear.summary()

## Training with fit method
linear.compile(optimizer="rmsprop",loss="mse",metrics=["mae"])
linear.fit(X,Y,batch_size = 8,epochs = 100)  

tf.print("w = ",linear.layers[1].kernel)
tf.print("b = ",linear.layers[1].bias)

## Save the model as pb format
export_path = "../data/linear_model/"
version = "1"       # Version could be used for management of further updates
linear.save(export_path+version, save_format="tf") 

```

```python

```

```python
!ls {export_path+version}
```

```python
# Check the info of the model file
!saved_model_cli show --dir {export_path+str(version)} --all
```

```python

```

The model file information marked red could be used later.

![](../data/模型文件信息.png)

```python

```

### 2. Create a spark (scala) project, insert jar package dependencies for TensorFlow in java.

```python

```

Need to add the following jar package dependency if use maven to manage projects.

```
<!-- https://mvnrepository.com/artifact/org.tensorflow/tensorflow -->
<dependency>
    <groupId>org.tensorflow</groupId>
    <artifactId>tensorflow</artifactId>
    <version>1.15.0</version>
</dependency>
```

You may also download the jar package `org.tensorflow.tensorflow`, together with the depended `org.tensorflow.libtensorflow` and `org.tensorflowlibtensorflow_jni` from the following link, then add all of them into the project.

https://mvnrepository.com/artifact/org.tensorflow/tensorflow/1.15.0


```python

```

```python

```

### 3. Loading TensorFlow model on the driver end of spark (scala) project and debug it successfully.


The following demonstration is run in jupyter notebook. We need to install toree to have it support spark(scala).

<!-- #region -->
```scala
import scala.collection.mutable.WrappedArray
import org.{tensorflow=>tf}

//Note: the second argument of the load function should be "serve"; the related info could be found from the model file.

val bundle = tf.SavedModelBundle 
   .load("/Users/liangyun/CodeFiles/eat_tensorflow2_in_30_days/data/linear_model/1","serve")

//Note: for the Java version TensorFlow uses static graph as TensorFlow 1.X, i.e. use `Session`, then explicit data to feed and results to fetch, and finally run it.
//Note: multiple feed methods could be used consequetively when we need to feed multiple data.
//Note: the input must be in the type of float

val sess = bundle.session()
val x = tf.Tensor.create(Array(Array(1.0f,2.0f),Array(2.0f,3.0f)))
val y =  sess.runner().feed("serving_default_inputs:0", x)
         .fetch("StatefulPartitionedCall:0").run().get(0)

val result = Array.ofDim[Float](y.shape()(0).toInt,y.shape()(1).toInt)
y.copyTo(result)

if(x != null) x.close()
if(y != null) y.close()
if(sess != null) sess.close()
if(bundle != null) bundle.close()  

result

```
<!-- #endregion -->

The output is:

```
Array(Array(3.019596), Array(3.9878292))
```


![](../data/TfDriver.png)

```python

```

### 4. Loading TensorFlow model on executor of spark (scala) project through RDD and debug it successfully


Here we transfer the TensorFlow model loaded on the Driver end to each executor through broadcasting, and predict with distributed computing on all the executors.


<!-- #region -->
```scala
import org.apache.spark.sql.SparkSession
import scala.collection.mutable.WrappedArray
import org.{tensorflow=>tf}

val spark = SparkSession
    .builder()
    .appName("TfRDD")
    .enableHiveSupport()
    .getOrCreate()

val sc = spark.sparkContext

// Loading model on Driver end
val bundle = tf.SavedModelBundle 
   .load("/Users/liangyun/CodeFiles/master_tensorflow2_in_20_hours/data/linear_model/1","serve")

// Broadcasting the model to all the executors
val broads = sc.broadcast(bundle)

// Creating dataset
val rdd_data = sc.makeRDD(List(Array(1.0f,2.0f),Array(3.0f,5.0f),Array(6.0f,7.0f),Array(8.0f,3.0f)))

// Predicting in batch by using the model through mapPartitions
val rdd_result = rdd_data.mapPartitions(iter => {
    
    val arr = iter.toArray
    val model = broads.value
    val sess = model.session()
    val x = tf.Tensor.create(arr)
    val y =  sess.runner().feed("serving_default_inputs:0", x)
             .fetch("StatefulPartitionedCall:0").run().get(0)

    // Copy the prediction into the Array in type Float with the same shape
    val result = Array.ofDim[Float](y.shape()(0).toInt,y.shape()(1).toInt)
    y.copyTo(result)
    result.iterator
    
})


rdd_result.take(5)
bundle.close
```
<!-- #endregion -->

```python

```

The output is:

```
Array(Array(3.019596), Array(3.9264367), Array(7.8607616), Array(15.974984))
```


![](../data/TfRDD.png)

```python

```

### 5. Loading TensorFlow model on executor of spark (scala) project through Data and debug it successfully


The distributed prediction using TensorFlow model could also be implemented on DataFrame data, besides implementing on RDD data in Spark.

It could be done through registering the method of prediction as a sparkSQL function.


<!-- #region -->
```scala
import org.apache.spark.sql.SparkSession
import scala.collection.mutable.WrappedArray
import org.{tensorflow=>tf}

object TfDataFrame extends Serializable{
    
    
    def main(args:Array[String]):Unit = {
        
        val spark = SparkSession
        .builder()
        .appName("TfDataFrame")
        .enableHiveSupport()
        .getOrCreate()
        val sc = spark.sparkContext
        
        
        import spark.implicits._

        val bundle = tf.SavedModelBundle 
           .load("/Users/liangyun/CodeFiles/master_tensorflow2_in_20_hours/data/linear_model/1","serve")

        val broads = sc.broadcast(bundle)
        
        // Construct the prediction function and register it as udf of sparkSQL
        val tfpredict = (features:WrappedArray[Float])  => {
            val bund = broads.value
            val sess = bund.session()
            val x = tf.Tensor.create(Array(features.toArray))
            val y =  sess.runner().feed("serving_default_inputs:0", x)
                     .fetch("StatefulPartitionedCall:0").run().get(0)
            val result = Array.ofDim[Float](y.shape()(0).toInt,y.shape()(1).toInt)
            y.copyTo(result)
            val y_pred = result(0)(0)
            y_pred
        }
        spark.udf.register("tfpredict",tfpredict)
        
        // Creating DataFrame dataset, and put the features into one of the columns
        val dfdata = sc.parallelize(List(Array(1.0f,2.0f),Array(3.0f,5.0f),Array(7.0f,8.0f))).toDF("features")
        dfdata.show 
        
        // Call the sparkSQL predicting function, add a new column as y_preds
        val dfresult = dfdata.selectExpr("features","tfpredict(features) as y_preds")
        dfresult.show 
        bundle.close
    }
}

```

<!-- #endregion -->

<!-- #region -->
```scala
TfDataFrame.main(Array())
```
<!-- #endregion -->

```
+----------+
|  features|
+----------+
|[1.0, 2.0]|
|[3.0, 5.0]|
|[7.0, 8.0]|
+----------+

+----------+---------+
|  features|  y_preds|
+----------+---------+
|[1.0, 2.0]| 3.019596|
|[3.0, 5.0]|3.9264367|
|[7.0, 8.0]| 8.828995|
+----------+---------+
```


We implemented distributed prediction using a linear regression model (implemented by `tf.keras`) using both RDD and DataFrame data structures in spark.

It is also possible to use the trained neural networks for distributed prediction through spark with just a slight modifications on this demonstration.

Actually the capability of TensorFlow is more than implementing neural networks, the low-level language of graph is able to express all kinds of numerical computation.

We are able to implement any kind of machine learning model on TensorFlow 2.0 with these various low-level APIs.

It is also possible to export the trained models as files and use it on the distributed system such as spark, which provides huge space of imagination for future applications.


Please leave comments in the WeChat official account "Python与算法之美" (Elegance of Python and Algorithms) if you want to communicate with the author about the content. The author will try best to reply given the limited time available.

You are also welcomed to join the group chat with the other readers through replying **加群 (join group)** in the WeChat official account.

![image.png](../data/Python与算法之美logo.jpg)
