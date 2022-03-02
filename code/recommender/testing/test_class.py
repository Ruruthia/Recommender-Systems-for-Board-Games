import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict


class TestClass(ABC):
    @abstractmethod
    def __init__(self, model):
        self.model = model
        self.errors = None
        self.top_n_df = None

    @abstractmethod
    def get_top_n(self, n=5, users_inner_id_subset=np.arange(1000)):
        """
        :param n: Number of recommendations to make.
        :param users_inner_id_subset: Array of inner ids of users to make recommendations for.
        :return: Dataframe with top n recommendations per user.
        """
        raise NotImplementedError()

    def coverage(self):
        """
        Given recommendations made by the model, return number of games recommended
        :return: Number of games recommended.
        """
        if self.top_n_df is None:
            self.top_n_df = self.get_top_n()
        recommended_games = self.top_n_df['bgg_id'].unique()

        return recommended_games.size

    def diversity(self, games_df, criterions=None):
        """
        Given recommendations made by the model and information about games,
        return the mean diversity of per user recommendations for each criterion.
        :param games_df: Dataframe of games, should contain 'bgg_id' column, and columns
        corresponding to each criterion containing list of item attributes.
        :param criterions: List of criterions, each should be present in games_df.
        :return: Dictionary of mean diversity per user recommendations for each criterion.
        """
        if self.top_n_df is None:
            self.top_n_df = self.get_top_n()
        if criterions is None:
            criterions = ['category', 'mechanic']
        games_df = games_df[['bgg_id'] + criterions].set_index('bgg_id')
        top_n_df = self.top_n_df[['bgg_user_name', 'bgg_id']]

        df = top_n_df.join(games_df, on='bgg_id', how='left')

        diversity_per_user = defaultdict(list)

        for _, user_df in df.groupby(by='bgg_user_name'):
            for criterion in criterions:
                user_criterion_diversity = np.unique(np.hstack(user_df[criterion].dropna())).size
                diversity_per_user[criterion].append(user_criterion_diversity)

        mean_diversity = {}
        for criterion in criterions:
            mean_diversity["diveristy_" + criterion] = np.mean(diversity_per_user[criterion])
        return mean_diversity

    def precision(self, implicit_test_df):
        """
        Calculate precision score (mean fraction of topN recommendations that is present in test set).
        :param implicit_test_df: Implicit test dataset. Must include columns 'bgg_user_name' and 'bgg_id'.
        :return: Mean precision score.
        """
        if self.top_n_df is None:
            self.top_n_df = self.get_top_n()
        precision = []
        top_n_series = self.top_n_df[['bgg_user_name', 'bgg_id']].groupby("bgg_user_name")["bgg_id"].apply(list)
        test_series = implicit_test_df[['bgg_user_name', 'bgg_id']].groupby("bgg_user_name")["bgg_id"].apply(list)
        df = top_n_series.to_frame(name='top_n').join(test_series.to_frame(name='test'), how='right')
        for _, row in df.iterrows():
            precision.append(np.in1d(row['top_n'], row['test']).mean())
        return sum(precision) / len(precision)

    def score(self, metric, **kwargs):
        """
        :param metric: Name of metric to calculate. Available metrics: diversity, coverage, precision.
        :param kwargs: Additional parameters.
        :return: Score for the specified metric.
        """
        metrics = {
            'diversity': self.diversity,
            'coverage': self.coverage,
            'precision': self.precision,
        }

        if metric in metrics:
            return metrics[metric](**kwargs)
        else:
            raise ValueError(f"Unknown metric {metric}.")


class ExplicitTests(TestClass):

    @abstractmethod
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def get_errors(self):
        raise NotImplementedError()

    def mse(self):
        """
        :return: Mean Square Error of the predicted ratings.
        """
        if self.errors is None:
            self.errors = self.get_errors()
        return np.mean(self.errors ** 2)

    def rmse(self):
        """
        :return:  Root Mean Square Error of the predicted ratings.
        """
        if self.errors is None:
            self.errors = self.get_errors()
        return np.sqrt(self.mse())

    def score(self, metric, **kwargs):
        """
        :param metric: Name of metric to calculate. Available metrics: diversity, coverage, precision, rmse, mse.
        :param kwargs: Additional parameters.
        :return: Score for the specified metric.
        """
        metrics = {
            'diversity': self.diversity,
            'coverage': self.coverage,
            'precision': self.precision,
            'rmse': self.rmse,
            'mse': self.mse,
        }

        if metric in metrics:
            return metrics[metric](**kwargs)
        else:
            raise ValueError(f"Unknown metric {metric}.")


class ImplicitTests(TestClass):

    @abstractmethod
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def precision_at_k(self):
        raise NotImplementedError()

    @abstractmethod
    def recall_at_k(self):
        raise NotImplementedError()

    @abstractmethod
    def auc_score(self):
        raise NotImplementedError()

    @abstractmethod
    def reciprocal_rank(self):
        raise NotImplementedError()

    def score(self, metric, **kwargs):
        """
        :param metric: Name of metric to calculate. Available metrics: diversity, coverage, precision,
         precision_at_k, recall_at_k, auc_score, reciprocal_rank.
        :param kwargs: Additional parameters.
        :return: Score for the specified metric.
        """
        metrics = {
            'diversity': self.diversity,
            'coverage': self.coverage,
            # We leave both precision and precision_at_k to allow usage of Lightfm implementation
            'precision': self.precision,
            'precision_at_k': self.precision_at_k,
            'recall_at_k': self.recall_at_k,
            'auc_score': self.auc_score,
            'reciprocal_rank': self.reciprocal_rank
        }

        if metric in metrics:
            return metrics[metric](**kwargs)
        else:
            raise ValueError(f"Unknown metric {metric}.")


def evaluate_model(metrics, test_class, games_df=None, diversity_criterions=None, implicit_test_df=None):
    """
    :param metrics: List of metrics' names.
    :param test_class: Appropriate TestClass
    :param games_df: Dataframe of games, should contain 'bgg_id' column, and columns
        corresponding to each criterion containing list of item attributes.
    :param diversity_criterions:  List of criterions, each should be present in games_df.
    :param implicit_test_df: Implicit test dataset. Must include columns 'bgg_user_name' and 'bgg_id'.
    :return: Dictionary of scores of specified metrics.
    """
    results = {}
    for metric in metrics:
        if metric == 'diversity':
            results = {**metrics, **test_class.score(metric, games_df=games_df, criterions=diversity_criterions)}
        if metric == 'precision':
            results[metric] = test_class.score(metric, implicit_test_df=implicit_test_df)
        else:
            results[metric] = test_class.score(metric)
    return results
