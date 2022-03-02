import numpy as np
import pandas as pd

"""We only need bgg_id, bgg_user_name, bgg_user_rating, bgg_user_owned columns from RatingItem,
 so we drop the rest.
 Additionally, we drop rows where rating is None and user does not own the game."""

batch_size = 1000000
n_iter = 0
slim_ratings_df = pd.DataFrame()
with open('data/bgg_RatingItem.jl') as f:
    batch = pd.read_json(f, lines=True, nrows=batch_size)
    while not batch.empty:
        print(f"Processing batch {n_iter}")
        slim_ratings_df = slim_ratings_df.append(
            batch[['bgg_id', 'bgg_user_name', 'bgg_user_rating', 'bgg_user_owned']])
        batch = pd.read_json(f, lines=True, nrows=batch_size)
        n_iter += 1
indices = np.any((slim_ratings_df['bgg_user_rating'].notna(), slim_ratings_df['bgg_user_owned'] == 1), axis=0)
slim_ratings_df = slim_ratings_df.loc[indices]
slim_ratings_df.to_csv('data/slimmed_ratings.csv.gz', compression='gzip', index=False)
