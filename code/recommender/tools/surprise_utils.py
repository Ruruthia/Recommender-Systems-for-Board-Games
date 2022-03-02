import numpy as np
import pandas as pd
import surprise
from tqdm.auto import tqdm

from recommender.testing.test_class import ExplicitTests

MODELS = {
    'SVD': surprise.SVD,
    'NMF': surprise.NMF,
}


class SurpriseTests(ExplicitTests):
    def __init__(self, model, test_df, trainset):
        super().__init__(model)
        self.test_df = test_df
        self.trainset = trainset

    def get_errors(self):
        errors = []
        ratings_df = self.test_df
        for _, rating in ratings_df.iterrows():
            est = self.model.predict(uid=rating['bgg_user_name'], iid=rating['bgg_id'])[3]
            err = est - rating['bgg_user_rating']
            errors.append(err)
        return np.array(errors)

    def _get_top_n_for_user(self, user_inner_id, user_name, n=5):
        user_items = np.array(self.trainset.ur[user_inner_id], dtype=int)[:, 0]
        items = np.setdiff1d(np.arange(self.trainset.n_items), user_items, assume_unique=True)
        user_anti_testset = [(user_name, self.trainset.to_raw_iid(i), 0) for
                             i in items]
        predictions = self.model.test(user_anti_testset)
        users_top_n = []
        for uid, iid, _, est, _ in predictions:
            users_top_n.append((uid, iid, est))
        users_top_n.sort(key=lambda x: x[2], reverse=True)
        return users_top_n[:n]

    def get_top_n_for_user_by_id(self, user_inner_id, n=5):
        user_name = self.trainset.to_raw_uid(user_inner_id)
        return self._get_top_n_for_user(user_inner_id=user_inner_id, user_name=user_name, n=n)

    def get_top_n_for_user_by_name(self, user_name, n=5):
        user_inner_id = self.trainset.to_inner_uid(user_name)
        return self._get_top_n_for_user(user_inner_id=user_inner_id, user_name=user_name, n=n)

    def get_top_n(self, n=5, users_inner_id_subset=np.arange(1000)):
        top_n = []
        for user_id in tqdm(users_inner_id_subset):
            top_n += self.get_top_n_for_user_by_id(user_inner_id=user_id, n=n)
        top_n_df = pd.DataFrame(top_n)
        top_n_df.columns = ['bgg_user_name', 'bgg_id', 'estimate']
        return top_n_df


def train_model(algo_name, params, trainset, verbose=False):
    model = MODELS[algo_name](**params, verbose=verbose)
    model.fit(trainset)

    return model
