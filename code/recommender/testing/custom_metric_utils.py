import numpy as np
import pandas as pd

from tqdm import tqdm


def to_1d(series):
    return pd.Series([x for _list in series if not isinstance(_list, float) for x in _list])


def extract_weights(series, func):
    if series.empty:
        return pd.Series(dtype=int)
    return series.explode().value_counts().map(func)


def get_players_count(games_features_df):
    players_count_df = pd.DataFrame({
        'min': games_features_df['min_players_best'].combine_first(games_features_df['min_players_rec']),
        'max': games_features_df['max_players_best'].combine_first(games_features_df['max_players_rec']),
    }).dropna().astype(int)

    return players_count_df.apply(lambda row: np.arange(row['min'], row['max'] + 1), axis=1)


def process_complexities(complexities, method, max_rating=5, num_buckets=21):
    buckets = np.arange(num_buckets) * max_rating / (num_buckets - 1)
    weights = np.zeros(num_buckets)

    if method == 'linear':
        for complexity in complexities:
            if not pd.isna(complexity):
                distances = np.abs(complexity - buckets)
                weights += np.maximum(1 - distances, 0)
        if weights.max() > 0:
            weights /= weights.max()

    elif method == 'nearest':
        for complexity in complexities:
            if not pd.isna(complexity):
                idx = int(np.rint(complexity * ((num_buckets - 1) / max_rating)))
                weights[idx] += 1
        if weights.sum() > 0:
            weights /= weights.sum()

    return pd.Series(weights, np.char.add("Complexity: ", buckets.astype(str)))


def create_profile_vector(
        games_df, categories_names, mechanics_names, max_players, normalize_fun, weight_fun,
        complexities_method
):
    weighted_categories = extract_weights(games_df['category'], weight_fun)
    weighted_categories = weighted_categories.reindex(categories_names, fill_value=0)
    if weighted_categories.sum() > 0:
        weighted_categories = normalize_fun(weighted_categories)

    weighted_mechanics = extract_weights(games_df['mechanic'], weight_fun)
    weighted_mechanics = weighted_mechanics.reindex(mechanics_names, fill_value=0)
    if weighted_mechanics.sum() > 0:
        weighted_mechanics = normalize_fun(weighted_mechanics)

    complexities = process_complexities(games_df['complexity'].to_numpy(), complexities_method)

    players_count = get_players_count(games_df)
    players_count = extract_weights(players_count, weight_fun)

    transformed_players_count = players_count.reindex(np.arange(1, max_players), fill_value=0)
    transformed_players_count.index = transformed_players_count.index.astype(str) + " players"
    transformed_players_count[f"{max_players}+ players"] = 0 \
        if len(players_count) == 0 \
        else players_count.where(players_count.index >= max_players).fillna(0).max()

    if transformed_players_count.sum() > 0:
        transformed_players_count = normalize_fun(transformed_players_count)

    return pd.concat((
        weighted_categories,
        weighted_mechanics,
        transformed_players_count,
        complexities,
    ))


def create_games_profiles_embeddings(
        games_features_df, category_names, mechanic_names, max_players, show_progress=False
):
    def profile_creation_fun(game):
        return pd.Series([game['bgg_id']], ['bgg_id']).append(
            create_profile_vector(
                pd.DataFrame([game]), category_names, mechanic_names, max_players, lambda weights: weights,
                lambda weights: weights, complexities_method='nearest',
            ),
        )

    if show_progress:
        tqdm.pandas()
        games_profiles_df = games_features_df.progress_apply(profile_creation_fun, axis=1)
    else:
        games_profiles_df = games_features_df.apply(profile_creation_fun, axis=1)
    return games_profiles_df.set_index('bgg_id')


def create_users_profiles_embeddings(
        train_df, games_features_df, categories_names, mechanics_names, max_players, users_subset=None,
        show_progress=False
):
    train_df = train_df[['bgg_user_name', 'bgg_id']]
    if users_subset is not None:
        train_df = train_df.loc[train_df['bgg_user_name'].isin(users_subset)]

    joined_df = train_df.join(games_features_df.set_index('bgg_id'), on='bgg_id')

    def profile_creation_fun(games_df):
        return create_profile_vector(
            games_df, categories_names, mechanics_names, max_players, lambda weights: weights / weights.sum(),
            weight_fun=lambda x: x, complexities_method='linear'
        )

    if show_progress:
        tqdm.pandas()
        users_profiles_df = joined_df.groupby('bgg_user_name').progress_apply(profile_creation_fun)
    else:
        users_profiles_df = joined_df.groupby('bgg_user_name').apply(profile_creation_fun)

    return users_profiles_df


def create_recommendations_profiles_embeddings(
        recommendations_df, num_recs, games_features_df, categories_names,
        mechanics_names, max_players, show_progress=False
):
    recommendations_df = recommendations_df[['bgg_user_name', 'bgg_id']]

    joined_df = recommendations_df.join(games_features_df.set_index('bgg_id'), on='bgg_id')

    def profile_creation_fun(games_df):
        return create_profile_vector(
            games_df, categories_names, mechanics_names, max_players, lambda weights: weights / num_recs,
            weight_fun=lambda x: x, complexities_method='nearest'
        )

    if show_progress:
        tqdm.pandas()
        recommendations_embeddings_df = joined_df.groupby('bgg_user_name').progress_apply(profile_creation_fun)
    else:
        recommendations_embeddings_df = joined_df.groupby('bgg_user_name').apply(profile_creation_fun)

    return recommendations_embeddings_df


def calculate_metric_scores(recommendation_profiles, users_profiles):
    recommendation_profiles, users_profiles = recommendation_profiles.sort_index(), users_profiles.sort_index()
    metric = recommendation_profiles.values * users_profiles.values
    c = metric[:, :20].sum(axis=1)
    m = metric[:, 20:40].sum(axis=1)
    p = metric[:, 40:50].sum(axis=1)
    cpx = metric[:, 50:].sum(axis=1)
    s = c + m + p + cpx
    return np.array([c.mean(axis=0), m.mean(axis=0), p.mean(axis=0), cpx.mean(axis=0), s.mean(axis=0)])
