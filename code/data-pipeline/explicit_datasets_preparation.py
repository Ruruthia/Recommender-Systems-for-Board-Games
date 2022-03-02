import pandas as pd
from recommender.testing.dataset_utils import split_ratings_dataset

"""For explicit dataset, we only include rows where rating is provided."""

columns = ['bgg_user_name', 'bgg_id', 'bgg_user_rating']
ratings_df = pd.read_csv('data/filtered_ratings.csv.gz', compression='gzip')

ratings_df = ratings_df[ratings_df['bgg_user_rating'] >= 1]
train_df, test_df = split_ratings_dataset(ratings_df)

train_df.to_csv('data/ratings_train_explicit.csv.gz', compression='gzip', index=False)
test_df.to_csv('data/ratings_test_explicit.csv.gz', compression='gzip', index=False)
ratings_df.to_csv('data/ratings_all_explicit.csv.gz', compression='gzip', index=False)
