from pyspark.sql import SparkSession, Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

spark = SparkSession \
    .builder \
    .appName("Recommender").getOrCreate()

lines = spark.read.text("../resources/train.csv").rdd
parts = lines.map(lambda row: row.value.split(","))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), itemId=int(p[1]),
                                     rating=int(p[2])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="itemId", ratingCol="rating")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

for prediction in predictions.take(10): 
    print(prediction)