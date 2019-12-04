import pandas as pd

# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel


def recommend(input_df):
    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    input_df['overview'] = input_df['overview'].fillna('')

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(input_df['overview'])

    # Output the shape of tfidf_matrix
    # print tfidf_matrix.shape

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Construct a reverse map of indices and movie titles
    indices = pd.Series(input_df.index, index=input_df['title_x']).drop_duplicates()

    movie_indices = get_recommendations('The Dark Knight Rises', cosine_sim, indices)

    recommendatations = input_df['title_x'].iloc[movie_indices]
    print recommendatations


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim, indices):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    return movie_indices
