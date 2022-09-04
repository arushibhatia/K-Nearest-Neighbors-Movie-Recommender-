import csv
import time

def run_code(movies, ratings, test_rating):
    """
    :param movie_name: txt with movie id and info about it
    :param ratings: csv with user ratings for movies
    :param test_rating:
    :return: tbd
    """

    # Set up movie_rating dict (key is movie ID, value is [ratings]
    # Set up user_rating dict (key is user ID, value is {movieID: rating}

    movie_ratings = {}
    user_ratings = {}

    # https://realpython.com/python-csv/

    with open(ratings, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if row[0] not in user_ratings.keys(): user_ratings[row[0]] = {}
            user_ratings[row[0]][row[1]] = int(row[2])

            if row[1] not in movie_ratings.keys(): movie_ratings[row[1]] = []
            movie_ratings[row[1]].append(int(row[2]))

            line_count += 1

    # Set up movie_names dict {id: name}

    movie_name = {}
    f = open(movies, encoding='ISO-8859-1')
    line = f.readline()

    while line:
        info = line.split('|')
        movie_name[info[0]] = info[1]
        line = f.readline()

    f.close()

    # Highest and Lowest Rating for 100+ User Ranking; Intalize the max and min values as (ranking, id)
    high = (0, '')
    low = (6, '')

    # Loop over all movies

    for m in movie_ratings.keys():
        # Num of Rankings
        total = len(movie_ratings[m])

        # if it has 100 ratings, consider
        if total >= 100:
            rating = sum(movie_ratings[m])/total

            # update highest and lowest, if necessary
            if rating > high[0]: high = (rating, m)
            if rating < low[0]: low = (rating, m)

    # Report values

    print("The highest rated movie with at least 100 ratings is %s (id %s) with a rating of %.2f"
          % (movie_name[high[1]], high[1], high[0]))

    print("The lowest rated movie with at least 100 ratings is %s (id %s) with a rating of %.2f"
          % (movie_name[low[1]], low[1], low[0]))

    # Find m_bar
    m_bar = 0

    # Loop over all users
    for u in user_ratings.keys():
        # Add the number of movies they have rated
        m_bar += len(user_ratings[u].keys())

    # Take the sum of ratings and divide by the user total and report the value
    m_bar = m_bar/len(user_ratings.keys())
    print("The average number of movie ratings by a user is %.2f" % m_bar)

    # print(len(user_ratings.keys()))

    # We have 3 dicts:
    # (user_ratings {user_id: {movie_id:rating}}, movie_ratings {movie_id:[ratings]}, movie_name {movie_id:movie_name}
    # user_id and movie_id and movie_name are strings; rating is an integer

    # P3
    # Set up parameters for recommendation; users is sorted so we can easily change range
    users = sorted(list(user_ratings.keys()), key=lambda x: (int(x), x))
    threshold = 3
    k = 30
    r = 5

    # User 1 movie recommendations
    movie_recommender('1', users, threshold, k, r, user_ratings, movie_name)

    # User 2 movie recommendations
    movie_recommender('2', users, threshold, k, r, user_ratings, movie_name)

    # To get the custom movie predictions, add the below lines to ratings.csv adn uncomment the code
    '''
    5555,50,5,1
    5555,172,5,1
    5555,182,4,1
    5555,237,4,1
    5555,763,4,1
    5555,820,4,1
    5555,523,4,1
    5555,71,4,1
    5555,69,3,1
    5555,56,3,1
    5555,1127,3,1
    5555,447,3,1
    5556,1000,2,879959583
    5556,20,5,879959584
    5556,18,2,879959585
    5556,934,1,879959586
    5556,231,2,879959587
    5556,938,4,879959588
    5556,291,3,879959589
    5556,119,2,879959590
    5556,837,5,879959591
    5556,234,1,879959592
    5556,435,3,879959593
    5556,384,4,879959594
    '''
    """
    Some thoughts:
    test_ratings is already an input, we just need to read it in (likely similar to how we do ratings
    movie_recommender calls helper functions so it should be easy to use just specific parts
    I think we should use a helper function to make the model and a helper function to test (so two separate)

    """
    print("\n4a")
    # Predicts scores yˆ for the first 1,000 user-item pairs on the first 1,000 lines of ratings.csv
    # Calculates and returns the RMSE based on those 1,000 ratings
    start_time = time.time()
    rmse = calculate_rmse(ratings, users, threshold, k, 3.5, user_ratings, False)
    end_time = time.time()
    total_time = end_time - start_time

    print("\nCalculated the RMSE based on predictions for the first 1000 user-item pairs in ratings.csv.")
    print("RMSE: " + str(rmse))
    print("Time to calculate RMSE: " + str(total_time) + " seconds.")

    # Calculates and returns the RMSE that predicts a rating of 3.5 for every pair
    start_time = time.time()
    rmse_baseline = calculate_rmse(ratings, users, threshold, k, 3.5, user_ratings, True)
    end_time = time.time()
    total_time = end_time - start_time
    print("\nCalculated the RMSE based on prediction defaulting to 3.5 for the first 1000 user-item pairs.")
    print("RMSE (BASELINE): " + str(rmse_baseline))
    print("Time to calculate RMSE: " + str(total_time) + " seconds.")

    print("\n4b")
    # Predicts scores yˆ for the pairs in test_ratings
    # Calculates and returns the RMSE based on predicted and actual ratings
    start_time = time.time()
    rmse = calculate_rmse('test_ratings.csv', users, threshold, k, 3.5, user_ratings, False)
    end_time = time.time()
    total_time = end_time - start_time

    print("\nCalculated the RMSE based on predictions based off of ratings.csv for test_ratings.csv pairs")
    print("RMSE: " + str(rmse))
    print("Time to calculate RMSE: " + str(total_time) + " seconds.")

    # Calculates and returns the RMSE that predicts a rating of 3.5 for every pair
    start_time = time.time()
    rmse_baseline = calculate_rmse( 'test_ratings.csv', users, threshold, k, 3.5, user_ratings, True)
    end_time = time.time()
    total_time = end_time - start_time
    print("\nCalculated the RMSE based on prediction defaulting to 3.5 for the first 1000 user-item pairs.")
    print("RMSE (BASELINE): " + str(rmse_baseline))
    print("Time to calculate RMSE: " + str(total_time) + " seconds.")


def calculate_rmse(testing_file, users, threshold, k, p, user_ratings, baseline):
    """
        This will calculate the scores for the first num_pairs_to_predict user-item pairs in ratings.csv
        :param testing_file: file that has all the test data, will help us determine which pairs we are testing for
        to calculate scores for, and compare against what the score should be
        :param users: set of users to consider, which is in this case, all users
        :param threshold: movies in common
        :param k: neighbors to use
        :param p: default rating
        :param user_ratings: dict {user_id: {movie_id:rating}}
        :param baseline: boolean that dictates whether or not we assume a prediction of p for all ratings
        :return: RMSE
        """

    rmse_sum = 0

    with open(testing_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            if (testing_file == 'ratings.csv' and row_count < 1000) or (testing_file != 'ratings.csv'):
                if baseline is False:
                    # finds k nearest neighbors for the first ID in the given row of the TESTING FILE based on data from
                    # training data
                    N = nearest_neighbor_search(row[0], users, threshold, k, user_ratings)

                    # we want to predict the score for this specific movie
                    M = [row[1]]

                    # predict rating for the specific movie (M) based on training data (user_ratings)
                    movie_rec_rating = smoothed_predictions(M, N, 3.5, user_ratings)
                    # print("predicted rating " + str(movie_rec_rating[row[1]]) + " actual rating " + row[2])

                    # calculate square distance from value in testing set and rating predicted from training data
                    rmse_sum += ((float(row[2]) - movie_rec_rating[row[1]]) ** 2)
                else:
                    rmse_sum += ((float(row[2]) - p) ** 2)
                row_count += 1

    rmse_sum = rmse_sum/float(row_count)
    rmse_final = rmse_sum ** 0.5
    return rmse_final


def movie_recommender(main_user, users, threshold, k, r, user_ratings, movie_name):

    """
    This will find the top r movie recommendations for the main user and report them to the console
    :param main_user: main user
    :param users: set of users to consider
    :param threshold: movies in common
    :param k: neighbors to use
    :param r: movies to return
    :param user_ratings: dict {user_id: {movie_id:rating}}
    :param movie_name: {movie_id:movie_name}
    :return: None
    """

    # find N and M
    N = nearest_neighbor_search(main_user, users, threshold, k, user_ratings)
    M = movie_set_finder(main_user, N, user_ratings)

    # get ratings for each movie
    movie_rec_ratings = smoothed_predictions(M, N, 3.5, user_ratings)

    # sort the list and get top r movies
    top_r_recommendations = sorted(movie_rec_ratings.keys(), key=movie_rec_ratings.get, reverse=True)


    # Report the values to the console
    print('\nThe top %d movies recommended for user %s using %d neighbors and a threshold of %d' %
          (r, main_user, k, threshold))

    # Loop over top r movies
    for i in range(r):
        # Get title and score and report to the console
        title = movie_name[top_r_recommendations[i]]
        score = movie_rec_ratings[top_r_recommendations[i]]
        print('%d. %s - Predicted Score: %.3f' % (i+1, title, score))


def nearest_neighbor_search(main_user, users, threshold, k, user_ratings):
    """
    A k nearest neighbor search for a given user
    :param main_user: main user for search
    :param users: list of users
    :param threshold: movies in common threshold
    :param k: top results to report
    :param user_ratings: dict {user_id: {movie_id:rating}}
    :return: top k neighbors
    """

    # dictionary to store top users

    user_neighbors = {}

    for user in users:

        # Do not check the same user
        if user == main_user: continue
        # Find distance
        user_neighbors[user] = find_angular_distance(user_ratings, main_user, user, threshold)

    # sort and report the top results
    return sorted(user_neighbors.keys(), key=user_neighbors.get)[:k]


def find_angular_distance(user_ratings, user1_id, user2_id, threshold):

    """
    This function will find the angular distance between two users
    :param user_ratings: dict {user_id: {movie_id:rating}}
    :param user1_id: first user id (str)
    :param user2_id: second user id (str)
    :param threshold: movies that need to have both rated
    :return:
    """

    # Determine the movies ratings they have in common
    movies_in_common = list(user_ratings[user1_id].keys() & user_ratings[user2_id].keys())
    num_in_common = len(movies_in_common)

    # If they do not meet threshold, return 1 as the distnace
    if threshold > num_in_common: return 1

    # Else they have met threshold
    else:
        # Initialize sums counters
        numerator_sum = 0
        denominator_x_sum = 0
        denominator_y_sum = 0

        # Loop over common movies
        for movie in movies_in_common:

            # Get ratings
            user1_rating = user_ratings[user1_id][movie]
            user2_rating = user_ratings[user2_id][movie]

            # Update sums based on the formula
            numerator_sum += user1_rating * user2_rating
            denominator_x_sum += user1_rating ** 2
            denominator_y_sum += user2_rating ** 2

        # Calculate the distance based on the formula and return the value
        xy_formula = numerator_sum / ((denominator_x_sum ** 0.5) * (denominator_y_sum ** 0.5))
        angular_distance = 1 - xy_formula
        return angular_distance


def movie_set_finder(main, users, user_ratings):
    """
    will find the set of movies that main has not seen but users has rated
    :param main: main user
    :param users: set of other users
    :param user_ratings: dict {user_id: {movie_id:rating}}
    :return: set of movies rated by users and not seen/rated by main
    """

    # Set for movies
    movies = set()

    # Get movies seen by main
    seen = set(user_ratings[main].keys())

    # Loop over users
    for user in users:
        # Loop over their movies
        for movie in user_ratings[user].keys():
            # Add if the movie has not been seen
            if movie not in seen: movies.add(movie)

    return movies


def smoothed_predictions(M, N, p, user_ratings):
    """
    this will return each movie in M with a smoothed rating
    :param M: set of movies
    :param N: user ratings
    :param p: default rating
    :param user_ratings: dict {user_id: {movie_id:rating}}
    :return:
    """

    # dict to output {movie: smoothed rating}
    ratings = {}

    # Loop over movies
    for movie in M:
        # set up N_j and y_j
        N_j = 0
        y_j = 0

        # Loop over users
        for user in N:
            # Only consider if they rated the movie
            if movie in user_ratings[user].keys():
                # Add to N_j and y_j
                N_j += 1
                y_j += user_ratings[user][movie]

        # If none of the nearest neighbors rated that movie, then the smooth prediction is just p
        if N_j != 0:
            # Update movie values with smoothed
            y_j = y_j/N_j
            smooth = (p + N_j*y_j)/(1 + N_j)
            ratings[movie] = smooth
        else:
            ratings[movie] = p

    return ratings


if __name__ == '__main__':

    run_code('movies.txt', 'ratings.csv', 'test_ratings.csv')

