import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score, reciprocal_rank, recall_at_k

from recommender.testing.test_class import ImplicitTests


class LightFMTests(ImplicitTests):
    def __init__(self, model, train_df, dataset, train_interactions, test_interactions, item_features):
        super().__init__(model)
        self.dataset = dataset
        self.train_interactions = train_interactions
        self.test_interactions = test_interactions
        self.item_features = item_features
        self.user_mapping = dataset.mapping()[0]
        self.games_mapping = dataset.mapping()[2]
        self.train_df = train_df
        self.num_games = len(self.games_mapping)

    def _get_top_n_for_user(self, user_inner_id, training_games_ids, n=5, num_threads=8):
        known_ids = [self.games_mapping[i] for i in training_games_ids]
        unknown_ids = np.array([i for i in range(self.num_games) if i not in known_ids])
        ratings = self.model.predict(user_inner_id, unknown_ids, item_features=self.item_features,
                                     num_threads=num_threads)
        games_ids = np.argsort(ratings)[::-1][:n]
        top_n = []
        for idx in games_ids:
            idx = unknown_ids[idx]
            top_n.append(
                list(self.games_mapping.keys())[list(self.games_mapping.values()).index(idx)]
            )
        return top_n

    def get_top_n_for_user_by_id(self, user_inner_id, training_games_ids, n=5, num_threads=8):
        return self._get_top_n_for_user(user_inner_id=user_inner_id, training_games_ids=training_games_ids, n=n,
                                        num_threads=num_threads)

    def get_top_n_for_user_by_name(self, user_name, training_games_ids, n=5, num_threads=8):
        user_inner_id = self.user_mapping[user_name]
        return self._get_top_n_for_user(user_inner_id=user_inner_id, training_games_ids=training_games_ids, n=n,
                                        num_threads=num_threads)

    def get_top_n(self, n=5, users_inner_id_subset=np.arange(1000), num_threads=8):
        bgg_user_names = np.array(list(self.dataset.mapping()[0].keys()))[users_inner_id_subset]
        recommendations = []
        training_df_grouped = self.train_df.groupby("bgg_user_name")["bgg_id"].apply(list)
        for user_inner_id, user_name in zip(users_inner_id_subset, bgg_user_names):
            try:
                training_games_ids = training_df_grouped[user_name]
            except KeyError:
                training_games_ids = []
            top_n = self.get_top_n_for_user_by_id(user_inner_id=int(user_inner_id), training_games_ids=training_games_ids,
                                                  n=n, num_threads=num_threads)
            recommendations.append({"bgg_user_name": user_name, "bgg_id": top_n})
        return pd.DataFrame(recommendations).explode('bgg_id').reset_index(drop=True)

    def train_precision_at_k(self, k=5, num_threads=8):
        return precision_at_k(
            self.model,
            test_interactions=self.train_interactions,
            item_features=self.item_features,
            k=k,
            num_threads=num_threads,
        ).mean()

    def precision_at_k(self, k=5, num_threads=8):
        return precision_at_k(
            self.model,
            test_interactions=self.test_interactions,
            train_interactions=self.train_interactions,
            item_features=self.item_features,
            k=k,
            num_threads=num_threads,
        ).mean()

    def recall_at_k(self, k=5, num_threads=8):
        return recall_at_k(
            self.model,
            test_interactions=self.test_interactions,
            train_interactions=self.train_interactions,
            item_features=self.item_features,
            k=k,
            num_threads=num_threads,
        ).mean()

    def auc_score(self, num_threads=8):
        return auc_score(
            self.model,
            test_interactions=self.test_interactions,
            train_interactions=self.train_interactions,
            item_features=self.item_features,
            num_threads=num_threads,
        ).mean()

    def reciprocal_rank(self, num_threads=8):
        return reciprocal_rank(
            self.model,
            test_interactions=self.test_interactions,
            train_interactions=self.train_interactions,
            item_features=self.item_features,
            num_threads=num_threads,
        ).mean()


def train_model(train_interactions, item_features, params, num_threads=8):
    epochs = params.pop('epochs')
    model = LightFM(**params)
    model.fit(train_interactions, verbose=True, item_features=item_features, epochs=epochs,
              num_threads=num_threads)
    return model
