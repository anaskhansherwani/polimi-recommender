from pyspark import SparkContext
import math, time

def main():
    # Create a SparkContext
    sc = SparkContext.getOrCreate()
    
    # Load and parse the data
    num_partitions = 10
    train_data = sc.textFile("../resources/train.csv", num_partitions)\
        .map(lambda l: l.split(','))
    test_data = sc.textFile("../resources/test.csv", num_partitions)\
        .map(lambda l: l.split(','))
    icm_data = sc.textFile("../resources/icm.csv", num_partitions)\
        .map(lambda l: l.split(','))

    ga = train_data.map(lambda x: float(x[2])).mean()
    ratings = train_data.map(lambda l: (int(l[0]), (int(l[1]), float(l[2]) - ga))).cache()

    no_of_items = icm_data.keys().distinct().count() 
    no_of_features = icm_data.values().distinct().count() 
    
    icm_item_count = dict(icm_data.keys()\
        .map(lambda w: (w, 1))\
        .reduceByKey(lambda x, y: x + y)\
        .map(lambda l: (l[0], math.sqrt(l[1])))\
        .collect())

    idf = dict(icm_data.values()\
        .map(lambda w: (w, 1))\
        .reduceByKey(lambda x, y: x + y)\
        .map(lambda l: (l[0], math.log(no_of_items/l[1], 10)))\
        .collect())

    icm_map = dict(icm_data.map(lambda l: (l[0] + "," + l[1], 1/icm_item_count.get(l[0])))\
        .collect())
    icm_item_count.clear()
    
    user_profile = []
    users = dict(test_data.map(lambda x: (int(x[0]), True)).collect())
    ratings_dict = dict(ratings.filter(lambda x: users.get(x[0], False))\
                        .groupByKey()\
                        .map(lambda x : (x[0], list(x[1])))\
                        .collect())
    count = 0
    time_counter = 0.0
    start = 0.0
    elapsed = 0.0
    while (count < no_of_features):
        start = time.time()
        print("Running for feature " + str(count) + " out of " + str(no_of_features))
        time_counter = time_counter + elapsed 
        time_average = time_counter / (count + 1)
        print("Estimated time remaining: " + str((time_average  * (no_of_features - count)) / 60))
        for each_user in users:
            user_ratings = ratings_dict.get(each_user, [])
            profile_value = 0
            for each_rating in user_ratings:
                profile_value = profile_value + (icm_map.get(str(each_rating[0]) + "," + str(count + 1), 0) * each_rating[1])
            if (profile_value != 0):
                user_profile.append((str(each_user) + "," + str(count + 1), profile_value))
        count = count + 1
        end = time.time()
        elapsed = (end - start)
        
    user_profile = dict(user_profile)
    new_ratings = []
    items = set(icm_data.keys().distinct().collect())
    items_with_features = dict(icm_data.groupByKey().map(lambda x: (x[0], list(x[1]))).collect())
    users_length = len(users)
    count = 0
    time_counter = 0.0
    start = 0.0
    elapsed = 0.0
    for each_user in users:
        start = time.time()
        print("Running for user " + str(count) + " out of " + str(users_length))
        time_counter = time_counter + elapsed 
        time_average = time_counter / (count + 1)
        print("Estimated time remaining: " + str((time_average  * (users_length - count)) / 60))
        user_ratings = ratings_dict.get(each_user, [])
        seen_items = set(map(lambda x: x[0], user_ratings))
        unseen_items = list(items - seen_items)      
        for each_item in unseen_items:
            final_rating = 0
            item_features = items_with_features.get(each_item, [])
            for each_feature in item_features:
                feature_score = icm_map.get(str(each_item) + "," + str(each_feature), 0)
                user_score = user_profile.get(str(each_user) + "," + str(each_feature), 0)
                idf_score = idf.get(str(each_feature), 0)
                final_rating = final_rating + (feature_score * user_score * idf_score)
            new_ratings.append((each_user, each_item, final_rating))
        count = count + 1
        end = time.time()
        elapsed = (end - start)
    
    rdd = sc.parallelize(new_ratings).cache()
    file = open("../resources/results.csv", "w")
    file.write("UserId,RecommendedItemIds\n")
    for each_user in users:
        recommendations = str(each_user) + "," + " "\
            .join(map(str, (r[1] for r in rdd.filter(lambda x: x[0] == each_user)\
                            .takeOrdered(5, lambda x: -x[2]))))
        file.write(recommendations + "\n")    

if __name__ == '__main__':
    main()