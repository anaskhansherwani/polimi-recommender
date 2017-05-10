from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating

# Create a SparkContext
sc = SparkContext.getOrCreate()

# Load and parse the data
num_partitions = 10
train_data = sc.textFile("../resources/train.csv", num_partitions)
test_data = sc.textFile("../resources/test.csv", num_partitions)

# Convert training data into Ratings RDD
ratings = train_data.map(lambda l: l.split(','))\
    .map(lambda l: Rating(int(l[0]), int(l[1]), int(l[2])))
 
# Build the recommendation model using Alternating Least Squares
rank = 10
num_iterations = 10
model = ALS.train(ratings, rank, num_iterations)

# Evaluate the model on training data
evaluation_data = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(evaluation_data).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
print("Mean Squared Error = " + str(MSE))

# Get the recommendations for test data and store in file
num_recommendations = 5
file = open("../resources/results.csv", "w")
file.write("UserId,RecommendedItemIds\n")
for user in test_data.collect():
    recommendations = user + "," + " " \
        .join(map(str, (i.product for i in model \
        .recommendProducts(int(user), num_recommendations))))
    file.write(recommendations + "\n")
file.close()
