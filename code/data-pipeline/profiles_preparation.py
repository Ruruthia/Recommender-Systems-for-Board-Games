import pickle

import pandas as pd

from recommender.testing.custom_metric_utils import create_users_profiles_embeddings, create_games_profiles_embeddings

full_df = pd.read_csv(f'data/ratings_all_implicit.csv.gz')
train_df = pd.read_csv('data/ratings_train_implicit.csv.gz')
test_df = pd.read_csv(f'data/ratings_test_implicit.csv.gz')
games_df = pd.read_json('data/bgg_GameItem.jl', lines=True)[[
    'name', 'bgg_id', 'mechanic', 'category', 'complexity', 'max_players_best', 'min_players_best', 'max_players_rec',
    'min_players_rec'
]]
features_names = pd.read_csv('data/game_features_names.csv.gz').values.flatten()

with open('data/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

mechanics_names = features_names[:20]
categories_names = features_names[20:40]

users_profiles = create_users_profiles_embeddings(
    train_df, games_df, categories_names, mechanics_names, 10, show_progress=True
)

games_profiles = create_games_profiles_embeddings(
    games_df, categories_names, mechanics_names, 10, show_progress=True
)

# We want to construct a profile from train and test interactions, but only for users present in the test set
test_df = full_df[full_df['bgg_user_name'].isin(test_df['bgg_user_name'].unique())]
test_user_profiles = create_users_profiles_embeddings(
    test_df, games_df, categories_names, mechanics_names, 10, show_progress=True
)

users_profiles.to_pickle('data/users_profiles.pkl')
games_profiles.to_pickle('data/games_profiles.pkl')
test_user_profiles.to_pickle('data/test_users_profiles.pkl')
