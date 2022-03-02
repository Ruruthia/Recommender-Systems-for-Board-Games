import numpy as np
import pandas as pd


def split_ratings_dataset(ratings_df, seed=None, frac_users=0.7, frac_items=0.7):
    """
    Split the ratings dataset by users into training and testing sets.
    :param ratings_df: Dataframe of ratings, should contain 'bgg_user_name' column.
    :param seed: Seed for numpy random number generator. (default None)
    :param frac_users: Determines what fraction of users should be used for training. (default 0.7)
    :param frac_items: Determines what fraction of testing user's items should be used for training. (default 0.7)
    :return: Training and testing set.
    """

    if seed is not None:
        np.random.seed(seed)

    users = ratings_df['bgg_user_name'].unique()
    np.random.shuffle(users)
    train_size = int(frac_users * users.shape[0])

    train_df = ratings_df[ratings_df['bgg_user_name'].isin(users[:train_size])]
    test_df = ratings_df[ratings_df['bgg_user_name'].isin(users[train_size:])]

    test_known_df, test_unknown_df = split_testing_set(test_df, seed=seed, frac=frac_items)

    return pd.concat((train_df, test_known_df)), test_unknown_df


def split_testing_set(test_df, seed=None, frac=0.7):
    """
    Split the testing dataset into known part, used for training the model,
    and obscured part, used for calculating accuracy.
    :param test_df: Dataframe of ratings, should contain 'bgg_user_name' column.
    :param seed: Seed for numpy random number generator. (default None)
    :param frac: Determines what fraction of dataset should be used for training. (default 0.8)
    :return: Known and obscured parts of testing set.
    """

    if seed is not None:
        np.random.seed(seed)

    grouped = test_df.groupby(by='bgg_user_name')
    test_known = []
    test_unknown = []
    for _, df in grouped:
        df_size = df.shape[0]

        known_size = int(round(frac * df_size))
        known_indices = np.random.choice(df_size, known_size, replace=False)
        known_data = df.iloc[known_indices]
        test_known.append(known_data)

        unknown_indices = np.setdiff1d(np.arange(df_size), known_indices)
        unknown_data = df.iloc[unknown_indices]
        test_unknown.append(unknown_data)

    return pd.concat(test_known), pd.concat(test_unknown)


def to_positive(ratings):
    """
    :param ratings: Pandas dataframe with columns [bgg_user_owned, bgg_user_rating]
    :return: Pandas dataframe nly with ratings considered to be positive
    """
    ratings = ratings.loc[(
            (ratings['bgg_user_owned'] == 1.0) & (ratings['bgg_user_rating'].isnull()) |
            (ratings['bgg_user_rating'] > 6))].copy()
    ratings['value'] = 1
    return ratings


def prepare_interactions(train_df, test_df, dataset):
    """
    Prepare the interactions matrices for training and testing and dataframe with known ratings (appearing in
    interactions for training).
    :param train_df: Dataframe of training ratings, should contain 'bgg_user_name' column.
    :param test_df: Dataframe of testing ratings, should contain 'bgg_user_name' column.
    :param dataset: LightFM dataset fitted to both train_df and test_df.
    :return: Training interactions and testing interactions.
    """

    print("Preparing training interactions")
    train_interactions = dataset.build_interactions(
        (
            (val["bgg_user_name"], val["bgg_id"], val["value"])
            for idx, val in train_df.iterrows()
        )
    )[1]
    print("Preparing testing interactions")
    test_interactions = dataset.build_interactions(
        (
            (val["bgg_user_name"], val["bgg_id"], val["value"])
            for idx, val in test_df.iterrows()
        )
    )[1]
    return train_interactions, test_interactions
