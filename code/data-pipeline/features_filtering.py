import pandas as pd

"""
As game features, we only use 20 most popular game mechanics, categories and designers (excluding designer "Uncredited").
We save file with their names and another file with mapping bgg_id - filtered features.
"""


def to_1d(series):
    return pd.Series([x for _list in series for x in _list])


games_features_df = pd.read_json('data/bgg_GameItem.jl', lines=True)[['bgg_id', 'mechanic', 'category', 'designer']]
df_mechanics = games_features_df.loc[games_features_df['mechanic'].notna()]
mechanics = to_1d(df_mechanics['mechanic']).value_counts()
df_categories = games_features_df.loc[games_features_df['category'].notna()]
categories = to_1d(df_categories['category']).value_counts()
df_designers = games_features_df.loc[games_features_df['designer'].notna()]
designers = to_1d(df_designers['designer']).value_counts()
# drop Uncredited designer
designers = designers.drop(labels=['(Uncredited):3'])
# 20 most popular game mechanics, categories and designers
features = mechanics.head(20).index.append(categories.head(20).index).append(designers.head(20).index)
features = pd.DataFrame(features)
features.to_csv('data/game_features_names.csv.gz', compression='gzip', index=False)

features = features.values.flatten()
games_features_df['features'] = [[] for r in range(len(games_features_df))]

for index, row in games_features_df.iterrows():
    if type(row['category']) is list:
        row['features'].extend([category for category in row['category'] if category in features])
    if type(row['mechanic']) is list:
        row['features'].extend([mechanic for mechanic in row['mechanic'] if mechanic in features])
    if type(row['designer']) is list:
        row['features'].extend([designer for designer in row['designer'] if designer in features])

# Mapping bgg_id - filtered features
games_features_df = games_features_df[['bgg_id', 'features']]
games_features_df.to_csv('data/game_features.csv.gz', compression='gzip', index=False)
