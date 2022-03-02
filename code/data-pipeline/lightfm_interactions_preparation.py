import pickle
from ast import literal_eval

import pandas as pd
import scipy.sparse
from lightfm.data import Dataset

from recommender.testing.dataset_utils import prepare_interactions

train_df = pd.read_csv('data/ratings_train_implicit.csv.gz')
test_df = pd.read_csv('data/ratings_test_implicit.csv.gz')
full_df = pd.read_csv('data/ratings_all_implicit.csv.gz')
features_names = pd.read_csv('data/game_features_names.csv.gz').values.flatten()
game_features = pd.read_csv('data/game_features.csv.gz')

dataset = Dataset()
print("Fitting dataset")
dataset.fit((x for x in full_df['bgg_user_name']), (x for x in full_df['bgg_id']),
            item_features=(x for x in features_names))
with open('data/dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f, -1)

print("Preparing item features")
game_features['features'] = game_features.features.apply(literal_eval)
games_list = full_df['bgg_id'].unique()
game_features = game_features.drop((game_features['bgg_id'])
                                   [~game_features['bgg_id'].isin(games_list)].index)
item_features = dataset.build_item_features((val['bgg_id'], val['features'] + [val['bgg_id']])
                                            for idx, val in game_features.iterrows())
with open('data/item_features.pkl', 'wb') as f:
    pickle.dump(item_features, f, -1)

train_interactions, test_interactions = prepare_interactions(train_df, test_df, dataset)
scipy.sparse.save_npz('data/train_interactions.npz', train_interactions)
scipy.sparse.save_npz('data/test_interactions.npz', test_interactions)
