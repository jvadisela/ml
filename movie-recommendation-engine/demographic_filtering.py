import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


# v is the number of votes for the movie
# m is the minimum votes required to be listed in the chart
# R is the average rating of the movie
# C is the mean vote across the whole report

def recommend(joinedDF):
    C = joinedDF['vote_average'].mean()
    print "mean of vote_average : ", C

    m = joinedDF['vote_count'].quantile(0.9)
    print "min num of votes : ", m

    filteredMoviesDF = joinedDF[joinedDF['vote_count'] >= m]

    print joinedDF.shape
    print filteredMoviesDF.shape

    filteredMoviesDF['score'] = filteredMoviesDF.apply(lambda row: __weighted_rating(row, m, C), axis=1)

    # Sort movies based on score calculated above
    filteredMoviesDF = filteredMoviesDF.sort_values('score', ascending=False)

    print "Columns in filteredMoviesDF dataset : ", list(filteredMoviesDF.columns)

    # Print the top 15 movies
    print filteredMoviesDF[['title_x', 'vote_count', 'vote_average', 'score']].head(5)

    pop = filteredMoviesDF.sort_values('popularity', ascending=False)

    plt.figure(figsize=(12, 6))

    plt.barh(pop['title_x'].head(10), pop['popularity'].head(10), align='center', color='skyblue')
    #plt.gca().invert_yaxis()
    plt.xlabel("Popularity")
    plt.title("Popular Movies")
    plt.show()

    print pop['overview'].head(5)


def __weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)
