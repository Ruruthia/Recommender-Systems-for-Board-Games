import pandas as pd
from recommender.testing.dataset_utils import to_positive, split_ratings_dataset

"""For implicit dataset, we only include rows where the interaction was positive."""

columns = ['bgg_user_name', 'bgg_id', 'bgg_user_rating', 'bgg_user_owned']
ratings_df = pd.read_csv('data/filtered_ratings.csv.gz', compression='gzip')

ratings_df = to_positive(ratings_df)
train_df, test_df = split_ratings_dataset(ratings_df)

train_df.to_csv('data/ratings_train_implicit.csv.gz', compression='gzip', index=False)
test_df.to_csv('data/ratings_test_implicit.csv.gz', compression='gzip', index=False)
ratings_df.to_csv('data/ratings_all_implicit.csv.gz', compression='gzip', index=False)
