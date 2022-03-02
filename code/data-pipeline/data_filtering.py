import pandas as pd

"""We filter out games with less than 1000 interactions and users with less than 10 interactions."""

# Games filtering
ratings_df = pd.read_csv('data/slimmed_ratings.csv.gz', compression='gzip')
cleaned_game_df = []
for game, game_df in ratings_df.groupby(by='bgg_id'):
    if game_df.shape[0] < 1000:
        continue
    cleaned_game_df.append(game_df[:])
ratings_df = pd.concat(cleaned_game_df)

# Users filtering
cleaned_user_df = []
for user, user_df in ratings_df.groupby(by='bgg_user_name'):
    if user_df.shape[0] < 10:
        continue
    cleaned_user_df.append(user_df[:])
ratings_df = pd.concat(cleaned_user_df)
ratings_df.to_csv('data/filtered_ratings.csv.gz', compression='gzip', index=False)
print(f"Number of games after filtering: {ratings_df['bgg_id'].value_counts().shape}")
print(f"Number of users after filtering: {ratings_df['bgg_user_name'].value_counts().shape}")
