from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import shutil

def global_average(ratings):
    ga = ratings.map(lambda x: float(x[2])).mean()
    print("The global average is {}".format(ga))
    return ga
    
def test_model(sc, model):
    # Load and parse the data
    num_partitions = 10
    test_data = sc.textFile("../resources/test.csv", num_partitions)
         
    # Recommendation parameters
    num_recommendations = 5
    
    # Open file to output data
    file = open("../resources/results.csv", "w")
    file.write("UserId,RecommendedItemIds\n")
    
    # Get the recommendations for test data and store in file
    for user in test_data.collect():
        recommendations = user + "," + " "\
            .join(map(str, (r.product for r in model\
            .recommendProducts(int(user), num_recommendations))))
        file.write(recommendations + "\n")
    
    # Close the file
    file.close()

def evaluate_model(training, test, model):
    # Recommendation parameters
    num_recommendations = 5
    filter_criteria = global_average(training)
    
    # Open files to output data
    input_file = open("../resources/input.csv", "w")
    result_file = open("../resources/evaluation.csv", "w")
    
    # Collect input data to be used for recommendations
    only_users = training.map(lambda x: x[0]).collect()
    input_data = test.filter(lambda r: r[0] in only_users)\
        .filter(lambda r: r[2] >= filter_criteria)\
        .map(lambda p: (p[0], p[1]))\
        .groupByKey()\
        .map(lambda p: (p[0], list(p[1])))\
        .collect()
    
    # Recommend products on test data and store both input and test data
    for record in input_data:
        input_file.write(",".join(map(str, (i for i in record[1]))) + "\n")
        recommendations = ",".join(map(str, (r.product for r in model
                                             .recommendProducts(int(record[0]), num_recommendations))))
        result_file.write(recommendations + "\n")
    
    # Close the files
    input_file.close()
    result_file.close()
    
def train_model(training):
    # Build the recommendation model using Alternating Least Squares
    rank = 50
    num_iterations = 10
    model = ALS.train(training, rank, num_iterations, lambda_ = 0.001)
    
    # Evaluate the model on training data
    evaluation_data = training.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(evaluation_data).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = training.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(MSE))
    
    return model

def unbiased_ratings(ratings):
    ga = global_average(ratings)
    ratings_unbiased = ratings.map(lambda x: (x[0], x[1], float(x[2]) - ga))
    return ratings_unbiased
    
def item_bias(ratings_unbiased):
    c = 10
    sum_and_count = ratings_unbiased.map(lambda x: (x[1], float(x[2]))).\
        aggregateByKey((0, 0), lambda x, y: (x[0] + y, x[1] + 1),
                          lambda x, y: (x[0] + y[0], x[1] + y[1]))
    item_bias = sum_and_count.mapValues(lambda x: x[0] / (x[1] + c))
    return item_bias
    
def user_bias(ratings_unbiased):
    c = 10
    sum_and_count = ratings_unbiased.map(lambda x: (x[0], float(x[2]))).\
        aggregateByKey((0, 0), lambda x, y: (x[0] + y, x[1] + 1),
                          lambda x, y: (x[0] + y[0], x[1] + y[1]))
    user_bias = sum_and_count.mapValues(lambda x: x[0] / (x[1] + c))
    return user_bias
    
def global_effects(ratings):
    ratings_unbiased = unbiased_ratings(ratings)
    i_bias = item_bias(ratings_unbiased).collectAsMap()
    u_bias = user_bias(ratings_unbiased).collectAsMap()
    ga = global_average(ratings)
    
    ratings = ratings.map(lambda l: Rating(l[0], l[1], ga + i_bias[l[1]] + u_bias[l[0]]))
    
    return ratings
    
    
def main():
    # Create a SparkContext
    sc = SparkContext.getOrCreate()
    
    # Load and parse the data
    num_partitions = 10
    train_data = sc.textFile("../resources/train.csv", num_partitions)
    
    # Convert training data into Ratings RDD
    ratings = train_data.map(lambda l: l.split(','))\
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    ratings = global_effects(ratings)
    training = ratings
#     (training, test) = ratings.randomSplit([0.8, 0.2])
    
    # Train the model
    model = train_model(training)
    
    # Evaluate the model and save the results for MAP evaluation
#     evaluate_model(training, test, model)

    # Test the model and save the results for final submission
    test_model(sc, model)
    
    # Save the model
#     shutil.rmtree('../resources/myCollaborativeFilter')
#     model.save(sc, "../resources/myCollaborativeFilter")

if __name__ == '__main__':
    main()
