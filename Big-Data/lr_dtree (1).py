# Databricks notebook source
# MAGIC %md ###1- Gerekli kütüphaneler import edilir

# COMMAND ----------

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
#hdfs(hadoop)-dbfs gfs rdd
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf



from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

# COMMAND ----------

# MAGIC %md ### 2- Spark clusteri kontrol edilir

# COMMAND ----------

spark

# COMMAND ----------

# MAGIC %md ###3- Sınıflandırma işlemi için AUDİ araba fiyatı veriseti eklenir

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/audi.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "audi_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `audi_csv`

# COMMAND ----------

# MAGIC %md ###4- Yüklenen veri kümesinin şemasını yazdırılması

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# Let's define some constants which we will use throughout this notebook
NUMERICAL_FEATURES = ["year", 
                      "mileage",
                      "tax",
                      "mpg",
                      "engineSize",
                      "price",
                      ]
CATEGORICAL_FEATURES = [
                         "model",
                         "transmission",
                        ]
TARGET_VARIABLE = "fuelType"

# COMMAND ----------

print("{:d} Numerical features = [{:s}]".format(len(NUMERICAL_FEATURES), ", ".join(["`{:s}`".format(nf) for nf in NUMERICAL_FEATURES])))
print("{:d} Categorical features = [{:s}]".format(len(CATEGORICAL_FEATURES), ", ".join(["`{:s}`".format(nf) for nf in CATEGORICAL_FEATURES])))
print("1 Target variable = `{:s}`".format(TARGET_VARIABLE))

# COMMAND ----------

# MAGIC %md ### 5- Veri kümesinin ilk 5 satırını görüntülenmesi

# COMMAND ----------

df.show(5)

# COMMAND ----------

# MAGIC %md ###6- Boş(Null) değerlerin kontrol edilmesi

# COMMAND ----------

for c in df.columns:
  print("`{:s}` satırındaki null değer sayısı = {:d}".format(c, df.where(col(c).isNull()).count()))

# COMMAND ----------

# MAGIC %md ###7- İstatiksel özelliklerin görüntülenmesi

# COMMAND ----------

df.describe().toPandas() #transpose işlemi yaparak tabloyu daha rahat okuyabiliyoruz.

# COMMAND ----------

# Grafik kütüphanelerini kullanabilmek için önce PySpark DataFrame'imizi Pandas DataFrame'e dönüştürmemiz gerekiyor
pdf = df.toPandas()

# COMMAND ----------

# Seaborn özelliklerini kullanarak bazı varsayılan çizim yapılandırmaları ayarlanır. matplotlib
sns.set_style("darkgrid")
sns.set_context("notebook", rc={"lines.linewidth": 2,
"xtick.labelsize":14,
"ytick.labelsize":14,
"axes.labelsize": 18,
"axes.titlesize": 20,
})


# COMMAND ----------

# MAGIC %md ##8- Veri Dağılımlarının Analizi: Sayısal Özellikler

# COMMAND ----------

# MAGIC %md ###Sayısal özelliklerin dağılımları

# COMMAND ----------

# Her sütunun değerlerinin dağılımının çizilmesi

n_rows = 6
n_cols = 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7,20))

for i,f in enumerate(NUMERICAL_FEATURES):
    _ = sns.distplot(pdf[f],
                    kde_kws={"color": "#ca0020", "lw": 1}, 
                    hist_kws={"histtype": "bar", "edgecolor": "k", "linewidth": 1,"alpha": 0.8, "color": "#92c5de"},
                    ax=axes[i]
                    )

fig.tight_layout(pad=1.5)



# COMMAND ----------

# MAGIC %md ### İkili regresyon grafikleri

# COMMAND ----------

_ = sns.pairplot(data=pdf, 
                 vars=sorted(NUMERICAL_FEATURES), 
                 hue=TARGET_VARIABLE, 
                 kind="reg",
                 diag_kind='hist',
                 diag_kws = {'alpha':0.55, 'bins':20},
                 markers=["s","X","+"]
                )

# COMMAND ----------

# MAGIC %md ##9- Veri Dağılımlarının Analizi: Kategorik Özellikler

# COMMAND ----------

# MAGIC %md ###Bireysel kategorik özelliklerin histogramları

# COMMAND ----------



# For categorical variables, 'countplot' is the way to go
# Create a Figure containing 3x3 subplots
n_rows = 2
n_cols = 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14,14))

for i,f in enumerate(sorted(CATEGORICAL_FEATURES)): 
    ax = sns.countplot(pdf[f], ax=axes[i])
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

fig.tight_layout(pad=1.5)



# COMMAND ----------

# MAGIC %md ### 2. Kategorik özellikler ile hedef değişken (yakıt türü) arasındaki ilişki

# COMMAND ----------

n_rows = 2
n_cols = 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14,14))

i = 0
for c in sorted(CATEGORICAL_FEATURES):
    tmp_data = pd.crosstab(pdf.loc[:, c], pdf[TARGET_VARIABLE])
    # pandas.crosstab returns an mxn table where m is the number of values for the first argument (x) 
    # and n for the second argument (y)
    # As the second argument is always `TARGET_VARIABLE` (i.e., `deposit`), n = 2 (`deposit` is binary!)
    # e.g., x = 'housing'; y = 'deposit'
    # the following apply is used to transform the crosstab into a "normalized" table as follows:
    # each entry in the table displays how the i-th categorical value of x (i.e., i-th row) is distributed across
    # all the possible values of y (i.e., Y/N)
    tmp_data = tmp_data.apply(lambda x: x/tmp_data.sum(axis=1))
    ax = tmp_data.plot.bar(stacked=True, color=['red','green','yellow'], grid=False, ax=axes[i], legend=True)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    i += 1

fig.tight_layout(pad=1.5)



# COMMAND ----------

# MAGIC %md ##10- Model Eğitimi

# COMMAND ----------

# MAGIC %md ### Önce veri setimizin dengeli  olup olmadığını kontrol edelim

# COMMAND ----------

df.groupBy(TARGET_VARIABLE).count().show()

# COMMAND ----------

# MAGIC %md ### Veri Kümesi Bölme: Eğitim ve Test Seti

# COMMAND ----------

# MAGIC %md Veri dönüşümlerini içeren herhangi bir ön işlemeye geçmeden önce, veri setimizi 2 bölüme ayıracağız:
# MAGIC 
# MAGIC * Eğitim seti (örneğin, toplam örnek sayısının %80'ini oluşturan);
# MAGIC 
# MAGIC * Test seti (örn., örneklerin kalan %20'sini oluşturan)

# COMMAND ----------

RANDOM_SEED = 42

# COMMAND ----------

train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)

# COMMAND ----------

print("Training set size: {:d} instances".format(train_df.count()))
print("Test set size: {:d} instances".format(test_df.count()))

# COMMAND ----------

# MAGIC %md Bundan sonra sadece eğitim seti kısmı üzerinde çalışacağız. Öğrenilen modelimizi değerlendirdiğimizde test seti tekrar devreye girecek.

# COMMAND ----------

# MAGIC %md ### One-Hot Encoding kullanarak Kategorik özelliklerin Sayısal'a dönüştürülmesi

# COMMAND ----------

# MAGIC %md #### StringIndexer Örnek

# COMMAND ----------

columns = ["language","count"]
data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000"),("Python", "500000"),]

# COMMAND ----------

sdf = spark.createDataFrame(data,columns)

# COMMAND ----------

sdf.show()

# COMMAND ----------

stringIndexer = StringIndexer(inputCol="language", outputCol="indexed", stringOrderType="frequencyDesc")
model = stringIndexer.fit(sdf)
sdf = model.transform(sdf)

# COMMAND ----------

sdf.show()

# COMMAND ----------

# MAGIC %md #### OneHotEncoding Pipeline

# COMMAND ----------

# This function is responsible to implement the pipeline above for transforming categorical features into numerical ones
def to_numerical(df, numerical_features, categorical_features, target_variable):

    """
    Args:
        - df: the input dataframe
        - numerical_features: the list of column names in `df` corresponding to numerical features
        - categorical_features: the list of column names in `df` corresponding to categorical features
        - target_variable: the column name in `df` corresponding to the target variable

    Return:
        - transformer: the pipeline of transformation fit to `df` (for future usage)
        - df_transformed: the dataframe transformed according to the pipeline
    """
    
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler


    # 1. Create a list of indexers, i.e., one for each categorical feature
    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c), handleInvalid="keep") for c in categorical_features]

    # 2. Create the one-hot encoder for the list of features just indexed (this encoder will keep any unseen label in the future)
    encoder = OneHotEncoder(inputCols=[indexer.getOutputCol() for indexer in indexers], 
                                    outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers], 
                                    handleInvalid="keep")

    # 3. Indexing the target column (i.e., transform it into 0/1) and rename it as "label"
    # Note that by default StringIndexer will assign the value `0` to the most frequent label, which in the case of `deposit` is `no`
    # As such, this nicely resembles the idea of having `deposit = 0` if no deposit is subscribed, or `deposit = 1` otherwise.
    label_indexer = StringIndexer(inputCol = target_variable, outputCol = "label")
    
    # 4. Assemble all the features (both one-hot-encoded categorical and numerical) into a single vector
    assembler = VectorAssembler(inputCols=encoder.getOutputCols() + numerical_features, outputCol="features")

    # 5. Populate the stages of the pipeline
    stages = indexers + [encoder] + [label_indexer] + [assembler]

    # 6. Setup the pipeline with the stages above
    pipeline = Pipeline(stages=stages)

    # 7. Transform the input dataframe accordingly
    transformer = pipeline.fit(df)
    df_transformed = transformer.transform(df)

    # 8. Eventually, return both the transformed dataframe and the transformer object for future transformations
    return transformer, df_transformed

# COMMAND ----------

oh_transformer, oh_train_df = to_numerical(train_df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_VARIABLE)

# COMMAND ----------

#md oh_train_df.show(10) # [0 0 1 0]
oh_train_df.orderBy('model', ascending=False).show(10)

# COMMAND ----------

train = oh_train_df.select(["model_indexed","features", "label"])

# COMMAND ----------

train.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md ##  Logistic Regression 

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression # This corresponds to LogisticRegressionWithLBFGS

# This setting corresponds to no regularization at all (i.e., both regParam=0 and elasticNetParam=0)
log_reg = LogisticRegression(featuresCol = "features", labelCol = "label", maxIter=100)
log_reg_model = log_reg.fit(train)



# COMMAND ----------

log_reg_model.coefficientMatrix

# COMMAND ----------

from pyspark.ml.linalg  import  DenseMatrix
DenseMatrix(2, 2, range(4), 1)

# COMMAND ----------

# Collect training summary
trainingSummary = log_reg_model.summary

# COMMAND ----------

accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"% (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

# COMMAND ----------

oh_test_df = oh_transformer.transform(test_df)

# COMMAND ----------

oh_test_df.show(5)

# COMMAND ----------

# Select `features` and `label` only
test = oh_test_df.select(["features", "label"])
test.show(5)



# COMMAND ----------

predictions = log_reg_model.transform(test)

# COMMAND ----------

predictions.select("features", "prediction", "label").show(10)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator()
print('Test Set Accuracy: {:.3f}'.format(evaluator.evaluate(predictions)))

# COMMAND ----------

# MAGIC %md ### Hiperparametre Optimizasyonu

# COMMAND ----------

# This function defines the general pipeline for logistic regression
def logistic_regression_pipeline(train, 
                                 numerical_features, 
                                 categorical_features, 
                                 target_variable, 
                                 with_std=True,
                                 with_mean=True,
                                 k_fold=2):

    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline
    
    # Configure a logistic regression pipeline, which consists of the following stages: 
    # 1) convert categorical features to numerical ones
    # 2) standardize feature values (optional)
    # ... add any other custom transformation here ...
    # n) fit a logistic regression model


    # 1.a Create a list of indexers, i.e., one for each categorical feature
    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c), handleInvalid="keep") for c in categorical_features]

    # 1.b Create the one-hot encoder for the list of features just indexed (this encoder will keep any unseen label in the future)
    encoder = OneHotEncoder(inputCols=[indexer.getOutputCol() for indexer in indexers], 
                                    outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers], 
                                    handleInvalid="keep")

    # 1.c Indexing the target column (i.e., transform it into 0/1) and rename it as "label"
    # Note that by default StringIndexer will assign the value `0` to the most frequent label, which in the case of `deposit` is `no`
    # As such, this nicely resembles the idea of having `deposit = 0` if no deposit is subscribed, or `deposit = 1` otherwise.
    label_indexer = StringIndexer(inputCol = target_variable, outputCol = "label")
    
    # 1.d Assemble all the features (both one-hot-encoded categorical and numerical) into a single vector
    assembler = VectorAssembler(inputCols=encoder.getOutputCols() + numerical_features, outputCol="features")

    # 2.a Create the StandardScaler
    # scaler = StandardScaler(inputCol=assembler.getOutputCol(), outputCol="std_"+assembler.getOutputCol(), withStd=with_std, withMean=with_mean)
    # ...

    # 3 Populate the stages of the pipeline with all the preprocessing steps
    stages = indexers + [encoder] + [label_indexer] + [assembler] # + [scaler] + ...

    # 4. Create the logistic regression transformer
    log_reg = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100) # change `featuresCol=std_features` if scaler is used

    # 5. Add the logistic regression transformer to the pipeline stages (i.e., the last one)
    stages += [log_reg]

    # 6. Set up the pipeline
    pipeline = Pipeline(stages=stages)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # With 3 values for log_reg.regParam ($\lambda$) and 3 values for log_reg.elasticNetParam ($\alpha$),
    # this grid will have 3 x 3 = 9 parameter settings for CrossValidator to choose from.
    param_grid = ParamGridBuilder()\
    .addGrid(log_reg.regParam, [0.0, 0.05, 0.1]) \
    .addGrid(log_reg.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()
    
    cross_val = CrossValidator(estimator=pipeline, 
                               estimatorParamMaps=param_grid,
                               evaluator = MulticlassClassificationEvaluator(),
                               numFolds=k_fold,
                               collectSubModels=True # this flag allows us to store ALL the models trained during k-fold cross validation
                               )

    # Run cross-validation, and choose the best set of parameters.
    cv_model = cross_val.fit(train)

    return cv_model

# COMMAND ----------

cv_model = logistic_regression_pipeline(train_df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_VARIABLE)

# COMMAND ----------

for i, acc in enumerate(cv_model.avgMetrics):
    print("Avg. Accuracy computed across k-fold cross validation for model setting #{:d}: {:.3f}".format(i+1, acc))


# COMMAND ----------

print("Best model according to k-fold cross validation: lambda=[{:.3f}]; alfa=[{:.3f}]".
      format(cv_model.bestModel.stages[-1]._java_obj.getRegParam(), 
             cv_model.bestModel.stages[-1]._java_obj.getElasticNetParam(),
             )
      )
print(cv_model.bestModel.stages[-1])

# COMMAND ----------

# MAGIC %md ###Tahminlerde bulunmak için çapraz doğrulamanın en iyi modelini kullanmak

# COMMAND ----------

# Make predictions on the test set (`cv_model` contains the best model according to the result of k-fold cross validation)
# `test_df` will follow exactly the same pipeline defined above, and already fit to `train_df`
test_predictions = cv_model.transform(test_df)

# COMMAND ----------

test_predictions.select("features", "prediction", "label").show(5)

# COMMAND ----------

def evaluate_model(predictions, metric="accuracy"):
    
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    evaluator = MulticlassClassificationEvaluator(metricName=metric)

    return evaluator.evaluate(predictions)


# COMMAND ----------

print("***** Test Set *****")
print("Accuracy: {:.3f}".format(evaluate_model(test_predictions)))
print("F1 measure: {:.3f}".format(evaluate_model(test_predictions, metric="f1")))
print("***** Test Set *****")



# COMMAND ----------

# MAGIC %md ## Decision Tree

# COMMAND ----------

# This function defines the general pipeline for logistic regression
def decision_tree_pipeline(train, 
                           numerical_features, 
                           categorical_features, 
                           target_variable, 
                           with_std=True,
                           with_mean=True,
                           k_fold=5):

    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.classification import DecisionTreeClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline

    # Configure a decision tree pipeline, which consists of the following stages: 

    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c), handleInvalid="keep") for c in categorical_features]

    # Indexing the target column (i.e., transform it into 0/1) and rename it as "label"
    # Note that by default StringIndexer will assign the value `0` to the most frequent label, which in the case of `deposit` is `no`
    # As such, this nicely resembles the idea of having `deposit = 0` if no deposit is subscribed, or `deposit = 1` otherwise.
    label_indexer = StringIndexer(inputCol = target_variable, outputCol = "label")
    
    # Assemble all the features (both one-hot-encoded categorical and numerical) into a single vector
    assembler = VectorAssembler(inputCols=[indexer.getOutputCol() for indexer in indexers] + numerical_features, outputCol="features")

    # Populate the stages of the pipeline with all the preprocessing steps
    stages = indexers + [label_indexer] + [assembler] # + ...

    # Create the decision tree transformer
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label") # change `featuresCol=std_features` if scaler is used

    # 5. Add the decision tree transformer to the pipeline stages (i.e., the last one)
    stages += [dt]

    # 6. Set up the pipeline
    pipeline = Pipeline(stages=stages)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # With 3 values for dt.maxDepth and 2 values for dt.impurity
    # this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    param_grid = ParamGridBuilder()\
    .addGrid(dt.maxDepth, [3, 5, 8]) \
    .addGrid(dt.impurity, ["gini", "entropy"]) \
    .build()
    
    cross_val = CrossValidator(estimator=pipeline, 
                               estimatorParamMaps=param_grid,
                               evaluator=MulticlassClassificationEvaluator(metricName="accuracy"), # default = "accuracy", alternatively "f1"
                               numFolds=k_fold,
                               collectSubModels=True # this flag allows us to store ALL the models trained during k-fold cross validation
                               )

    # Run cross-validation, and choose the best set of parameters.
    cv_model = cross_val.fit(train)

    return cv_model

# COMMAND ----------

cv_model = decision_tree_pipeline(train_df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_VARIABLE)

# COMMAND ----------

for i, acc in enumerate(cv_model.avgMetrics):
    print("Avg. Accuracy computed across k-fold cross validation for model setting #{:d}: {:.3f}".format(i+1, acc))

# COMMAND ----------

print("Best model according to k-fold cross validation: maxDept=[{:d}]; impurity=[{:s}]".
      format(cv_model.bestModel.stages[-1]._java_obj.getMaxDepth(), 
             cv_model.bestModel.stages[-1]._java_obj.getImpurity(),
             )
      )
print(cv_model.bestModel.stages[-1])

# COMMAND ----------

test_predictions = cv_model.transform(test_df)

# COMMAND ----------

test_predictions.select("features", "prediction", "label").show(5)

# COMMAND ----------

print("***** Test Set *****")
print("Accuracy: {:.3f}".format(evaluate_model(test_predictions)))
print("F1 measure: {:.3f}".format(evaluate_model(test_predictions, metric="f1")))
print("***** Test Set *****")

# COMMAND ----------

def random_forest_pipeline(train, 
                           numerical_features, 
                           categorical_features, 
                           target_variable, 
                           with_std=True,
                           with_mean=True,
                           k_fold=5):

    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline

    # Configure a random forest pipeline, which consists of the following stages: 

    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c), handleInvalid="keep") for c in categorical_features]

    # Indexing the target column (i.e., transform it into 0/1) and rename it as "label"
    # Note that by default StringIndexer will assign the value `0` to the most frequent label, which in the case of `deposit` is `no`
    # As such, this nicely resembles the idea of having `deposit = 0` if no deposit is subscribed, or `deposit = 1` otherwise.
    label_indexer = StringIndexer(inputCol = target_variable, outputCol = "label")
    
    # Assemble all the features (both one-hot-encoded categorical and numerical) into a single vector
    assembler = VectorAssembler(inputCols=[indexer.getOutputCol() for indexer in indexers] + numerical_features, outputCol="features")

    # Populate the stages of the pipeline with all the preprocessing steps
    stages = indexers + [label_indexer] + [assembler] # + ...

    # Create the random forest transformer
    rf = RandomForestClassifier(featuresCol="features", labelCol="label") # change `featuresCol=std_features` if scaler is used

    # 5. Add the random forest transformer to the pipeline stages (i.e., the last one)
    stages += [rf]

    # 6. Set up the pipeline
    pipeline = Pipeline(stages=stages)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # With 3 values for rf.maxDepth and 3 values for rf.numTrees
    # this grid will have 3 x 3 = 9 parameter settings for CrossValidator to choose from.
    param_grid = ParamGridBuilder()\
    .addGrid(rf.maxDepth, [3, 5, 8]) \
    .addGrid(rf.numTrees, [10, 50, 100]) \
    .build()
    
    cross_val = CrossValidator(estimator=pipeline, 
                               estimatorParamMaps=param_grid,
                               evaluator=MulticlassClassificationEvaluator(metricName="accuracy"), # default = "areaUnderROC", alternatively "areaUnderPR"
                               numFolds=k_fold,
                               collectSubModels=True # this flag allows us to store ALL the models trained during k-fold cross validation
                               )

    # Run cross-validation, and choose the best set of parameters.
    cv_model = cross_val.fit(train)

    return cv_model

# COMMAND ----------

cv_model = random_forest_pipeline(train_df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_VARIABLE)

# COMMAND ----------

for i, acc in enumerate(cv_model.avgMetrics):
    print("Avg. Accuracy computed across k-fold cross validation for model setting #{:d}: {:.3f}".format(i+1, acc))

# COMMAND ----------

print("Best model according to k-fold cross validation: maxDept=[{:d}]".
      format(cv_model.bestModel.stages[-1]._java_obj.getMaxDepth(), 
             )
      )
print(cv_model.bestModel.stages[-1])

# COMMAND ----------

test_predictions = cv_model.transform(test_df)
test_predictions.select("features", "prediction", "label").show(5)

# COMMAND ----------

print("***** Test Set *****")
print("Accuracy: {:.3f}".format(evaluate_model(test_predictions)))
print("F1 measure: {:.3f}".format(evaluate_model(test_predictions, metric="f1")))
print("***** Test Set *****")
