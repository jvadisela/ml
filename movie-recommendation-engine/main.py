import matplotlib
import pandas as pd

import demographic_filtering
import content_based_filtering

matplotlib.use('TkAgg')

creditsDF = pd.read_csv('input/tmdb_5000_credits.csv')
moviesDF = pd.read_csv('input/tmdb_5000_movies.csv')

print "Columns in credits dataset : ", list(creditsDF.columns)
print "Columns in movies dataset : ", list(moviesDF.columns)

creditsDF.columns = ['id', 'title', 'cast', 'crew']
joinedDF = moviesDF.merge(creditsDF, on='id')

print joinedDF.head(5)


content_based_filtering.recommend(joinedDF)
