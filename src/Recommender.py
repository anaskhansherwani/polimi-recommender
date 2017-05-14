from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import shutil

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
    filter_criteria = 5
    
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
    rank = 10
    num_iterations = 10
    model = ALS.train(training, rank, num_iterations)
    
    # Evaluate the model on training data
    evaluation_data = training.map(lambda p: (p[0], p[1]))
    predictions = model.predictAll(evaluation_data).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = training.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(MSE))
    
    return model
    
def main():
    # Create a SparkContext
    sc = SparkContext.getOrCreate()
    
    # Load and parse the data
    num_partitions = 10
    train_data = sc.textFile("../resources/train.csv", num_partitions)
    
    # Convert training data into Ratings RDD
    ratings = train_data.map(lambda l: l.split(','))\
        .map(lambda l: Rating(int(l[0]), int(l[1]), int(l[2])))
    training = ratings
#     (training, test) = ratings.randomSplit([0.8, 0.2])
    
    # Train the model
    model = train_model(training)
    
    # Evaluate the model and save the results for MAP evaluation
#     evaluate_model(training, test, model)

    # Test the model and save the results for final submission
    test_model(sc, model)
    
    # Save the model
    shutil.rmtree('../resources/myCollaborativeFilter')
    model.save(sc, "../resources/myCollaborativeFilter")

if __name__ == '__main__':
    main()