"""
Collection of various Recommender Algorithm Implementations
"""
__author__ = "Marcel Kurovski"
__copyright__ = "Marcel Kurovski"
__license__ = "mit"

from collections import OrderedDict
from itertools import combinations
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .utils import get_entity_sim, sigmoid, setup_logging


setup_logging(logging.INFO)
_logger = logging.getLogger(__name__)


class FM(object):
    """
    Factorization Machine for regression, trained with row-wise SGD on the
    squared error.

    Prediction:
        y_hat(x) = w0 + sum_i w_i * x_i
                      + 0.5 * sum_f ( (sum_i v_{i,f} * x_i)^2
                                      - sum_i v_{i,f}^2 * x_i^2 )

    Drop-in replacement for `pylibfm.FM(task='regression', ...)`.
    """

    def __init__(self, num_factors: int = 8, num_iter: int = 10,
                 learning_rate: float = 0.001, init_stdev: float = 0.01,
                 seed: int = 42, verbose: bool = False):
        self.num_factors = num_factors
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.init_stdev = init_stdev
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        X = sp.csr_matrix(X)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.seed)

        self.w0_ = 0.0
        self.w_ = np.zeros(n_features)
        self.V_ = rng.normal(0.0, self.init_stdev, (n_features, self.num_factors))

        lr = self.learning_rate
        for epoch in range(self.num_iter):
            order = rng.permutation(n_samples)
            for i in order:
                start, end = X.indptr[i], X.indptr[i + 1]
                idx = X.indices[start:end]
                vals = X.data[start:end]

                v_sub = self.V_[idx]                 # (nnz, k)
                vx = v_sub * vals[:, None]           # (nnz, k)
                sum_vx = vx.sum(axis=0)              # (k,)
                sum_vx2 = (vx * vx).sum(axis=0)      # (k,)

                linear = self.w0_ + np.dot(self.w_[idx], vals)
                interaction = 0.5 * float((sum_vx * sum_vx - sum_vx2).sum())
                pred = linear + interaction

                err = pred - y[i]
                self.w0_ -= lr * 2.0 * err
                self.w_[idx] -= lr * 2.0 * err * vals
                grad_V = vals[:, None] * (sum_vx[None, :] - v_sub * vals[:, None])
                self.V_[idx] -= lr * 2.0 * err * grad_V

            if self.verbose:
                rmse = float(np.sqrt(np.mean((self.predict(X) - y) ** 2)))
                print(f"Epoch {epoch + 1:02d}: train RMSE={rmse:.4f}")
        return self

    def predict(self, X) -> np.ndarray:
        X = sp.csr_matrix(X)
        linear = self.w0_ + np.asarray(X @ self.w_).ravel()
        XV = X @ self.V_
        X2V2 = X.multiply(X) @ (self.V_ * self.V_)
        interaction = 0.5 * np.asarray(XV * XV - X2V2).sum(axis=1)
        return np.asarray(linear + interaction).ravel()


class BPRRecommender(object):
    def __init__(self, ratings: pd.DataFrame, users: np.array, items: np.array,
                 k: int, N: int, seed: int = 42):
        self.ratings = ratings
        self.n_ratings = len(ratings)
        self.users = sorted(users)
        self.items = np.array(sorted(items))
        self.m = len(self.users)
        self.n = len(self.items)
        self.k = k
        self.N = N
        self.seed = seed

        self.user_pos_items = {}
        self.user_neg_items = {}
        self.user_factors = None
        self.item_factors = None

        self._setup()

    def _setup(self):
        random_state = np.random.RandomState(self.seed)

        # Latent Factor initialization according to LightFM
        self.user_factors = (random_state.rand(self.m, self.k) - 0.5) / self.k
        self.item_factors = (random_state.rand(self.n, self.k) - 0.5) / self.k

        # Shuffle Ratings to break user-item contingency in SGD updates
        # and thus avoid slow convergence
        self.ratings = self.ratings.sample(frac=1, random_state=self.seed)

        self._init_user_items()

    def _init_user_items(self):
        grouped = self.ratings[['user', 'item']].groupby('user')
        groups = grouped.groups.keys()
        for user in self.users:
            pos_items = []
            if user in groups:
                pos_items = grouped.get_group(user).item.values
            neg_items = np.setdiff1d(self.items, pos_items)
            self.user_pos_items[user] = pos_items
            self.user_neg_items[user] = neg_items

    # TODO: Shorten
    def train(self, epochs: int, learning_rate: float, l2_decay: Dict[str, float] = None,
              verbose: bool = False):
        ratings_arr = self.ratings.values
        if l2_decay is None:
            l2_decay = {'user': 0.0, 'pos': 0.0, 'neg': 0.0}
            _logger.info("No L2 Decay regularization set")
        else:
            _logger.info(f"L2 Decay regularization (user, pos, neg): {l2_decay}")

        for epoch in range(epochs):

            for _ in range(len(self.ratings)):
                random_index = np.random.randint(self.n_ratings)
                user, pos_item = tuple(ratings_arr[random_index])
                neg_item = self._negative_sampling(user)

                # TODO: Align indices by making mapping user ids to compact and 0-indexed space
                # Deduct 1 as user ids are 1-indexed, but array is 0-indexed
                user_embed = self.user_factors[user - 1]
                pos_item_embed = self.item_factors[pos_item - 1]
                neg_item_embed = self.item_factors[neg_item - 1]

                user_grad, pos_item_grad, neg_item_grad = \
                    self._compute_gradients(user_embed,
                                            pos_item_embed,
                                            neg_item_embed,
                                            l2_decay)

                # update
                self.user_factors[user - 1] -= learning_rate * user_grad
                self.item_factors[pos_item - 1] -= learning_rate * pos_item_grad
                self.item_factors[neg_item - 1] -= learning_rate * neg_item_grad

            if verbose:
                samples = ratings_arr[-1000:]
                self._print_update(epoch, samples)

    def _print_update(self, epoch: int, samples: np.array) -> float:
        # take the 1000 most recent ratings and compute the mean ranking loss
        users = samples[:, 0]
        pos_items = samples[:, 1]
        neg_items = np.array([self._negative_sampling(user)
                              for user in users])

        user_embeds = self.user_factors[users - 1]
        pos_item_embeds = self.item_factors[pos_items - 1]
        neg_item_embeds = self.item_factors[neg_items - 1]

        pos_preds = np.sum(user_embeds * pos_item_embeds, axis=1)
        neg_preds = np.sum(user_embeds * neg_item_embeds, axis=1)
        preds = pos_preds - neg_preds

        loss = -np.log(sigmoid(preds)).mean()
        print(f"Epoch {epoch+1:02d}: Mean Ranking Loss: {loss:.4f}")

        return loss

    def _update_latent_factors(self):
        return None

    def _negative_sampling(self, user: int) -> int:
        """
        Return the item ids for negative samples
        """
        # TODO: Allow for popularity-based pdf for choosing negative item

        return np.random.choice(self.user_neg_items[user])

    @staticmethod
    def _compute_gradients(user_embed: np.array,
                           pos_item_embed: np.array,
                           neg_item_embed: np.array,
                           l2_decay: Dict[str, float]) -> Tuple[np.array, np.array, np.array]:
        """
        """
        pos_pred = np.sum(user_embed * pos_item_embed)
        neg_pred = np.sum(user_embed * neg_item_embed)
        pred = pos_pred - neg_pred

        generic_grad = (-np.exp(-pred) * sigmoid(pred))

        user_grad = generic_grad * (pos_item_embed - neg_item_embed)
        pos_item_grad = generic_grad * user_embed
        neg_item_grad = generic_grad * (-user_embed)

        user_grad += user_embed * l2_decay['user']
        pos_item_grad += pos_item_embed * l2_decay['pos']
        neg_item_grad += neg_item_embed * l2_decay['neg']

        return user_grad, pos_item_grad, neg_item_grad

    def get_recommendations(self, user: int, remove_known_pos: bool = False) -> List[Tuple[int, Dict[str, float]]]:
        predictions = self.get_prediction(user, remove_known_pos=remove_known_pos)
        recommendations = []
        # TODO: Simplify
        for item, pred in predictions.items():
            add_item = (item, pred)
            recommendations.append(add_item)
            if len(recommendations) == self.N:
                break

        return recommendations

    def get_prediction(self, user: int, items: np.array = None, remove_known_pos: bool = False) -> Dict[int, Dict[str, float]]:
        if items is None:
            if remove_known_pos:
                # Predict from unobserved items
                items = self.user_neg_items[user]
            else:
                items = np.array(self.items)
        if type(items) == np.int64:
            items = np.array([items])

        user_embed = self.user_factors[user - 1].reshape(1, -1)
        item_embeds = self.item_factors[items - 1].reshape(len(items), -1)

        # use array-broadcasting
        preds = np.sum(user_embed * item_embeds, axis=1)
        sorting = np.argsort(preds)[::-1]
        preds = {item: {'pred': pred} for item, pred in
                 zip(items[sorting], preds[sorting])}

        return preds


class MFRecommender(object):
    def __init__(self, ratings: pd.DataFrame, users: np.array, items: np.array,
                 k: int, N: int, seed: int = 42):
        self.ratings = ratings
        self.users = sorted(users)
        self.items = sorted(items)
        self.m = len(self.users)
        self.n = len(self.items)
        self.k = k
        self.N = N
        self.user_ratings = {}
        self.setup(seed)

    def setup(self, seed: int = 42):
        np.random.seed(seed)
        self.user_factors = np.random.normal(0, 1, (self.m, self.k))
        self.item_factors = np.random.normal(0, 1, (self.n, self.k))
        self.ratings = self.ratings.sample(frac=1, random_state=seed)

        grouped = self.ratings[['user', 'item', 'rating']].groupby('user')
        for user in self.users:
            vals = grouped.get_group(user)[['item', 'rating']].values
            self.user_ratings[user] = dict(zip(vals[:, 0].astype(int),
                                               vals[:, 1].astype(float)))

    # TODO: Implement weight decay regularization
    def train(self, epochs: int, batch_size: int, learning_rate: float) -> List[float]:
        num_batches = int(np.ceil(len(self.ratings) / batch_size))
        rmse_trace = []
        for epoch in range(epochs):
            for idx in range(num_batches):
                minibatch = self.ratings.iloc[idx * batch_size:(idx + 1) * batch_size][
                    ['user', 'item', 'rating']]
                # deduct 1 as user ids are 1-indexed, but array is 0-indexed
                user_embeds = self.user_factors[minibatch['user'].values - 1]
                item_embeds = self.item_factors[minibatch['item'].values - 1]

                user_grads, item_grads = self.compute_gradients(minibatch['rating'].values,
                                                                user_embeds,
                                                                item_embeds)

                self.user_factors[
                    minibatch['user'].values - 1] -= learning_rate * user_grads
                self.item_factors[
                    minibatch['item'].values - 1] -= learning_rate * item_grads

                if not idx % 50:
                    rmse = (self.rmse(minibatch['rating'].values,
                                      user_embeds,
                                      item_embeds))
                    rmse_trace.append(rmse)
                    print(f"{rmse:.3f}")

        return rmse_trace

    def get_recommendations(self, user: int) -> List[Tuple[int, Dict[str, float]]]:
        known_items = list(self.user_ratings[user].keys())
        predictions = self.get_prediction(user)
        recommendations = []
        for item, pred in predictions.items():
            if item not in known_items:
                add_item = (item, pred)
                recommendations.append(add_item)
            if len(recommendations) == self.N:
                break

        return recommendations

    def get_prediction(self, user: int, items: np.array = None, remove_known_pos: bool = False) -> Dict[int, Dict[str, float]]:
        if items is None:
            if remove_known_pos:
                # Predict from unobserved items
                items = self.user_neg_items[user]
            else:
                items = np.array(self.items)
        if type(items) == np.int64:
            items = np.array([items])

        user_embed = self.user_factors[user - 1].reshape(1, -1)
        item_embeds = self.item_factors[items - 1].reshape(len(items), -1)

        # use array-broadcasting
        preds = np.sum(user_embed * item_embeds, axis=1)
        sorting = np.argsort(preds)[::-1]
        preds = {item: {'pred': pred} for item, pred in
                 zip(items[sorting], preds[sorting])}

        return preds

    @staticmethod
    def compute_gradients(rating: float, u: np.array, v: np.array) -> Tuple[np.array, np.array]:
        # TODO: also test for stochastic case the shape to be 2-D
        pred = np.sum(u * v, axis=1)
        error = (rating - pred).reshape(-1, 1)

        u_grad = -2 * error * v
        v_grad = -2 * error * u

        return u_grad, v_grad

    @staticmethod
    def rmse(rating, u, v) -> float:
        pred = np.sum(u * v, axis=1)
        error = rating - pred
        rmse = np.sqrt(np.mean(error ** 2))
        return rmse


class NearestNeighborRecommender(object):
    """
    Implementation of user-based, neighborhood-based collaborative filtering
    """
    def __init__(self,
                 ratings: pd.DataFrame,
                 users: np.array,
                 items: np.array,
                 k: int = 10,
                 N: int = 5,
                 C: int = 1,
                 metric: str = 'pearson'):
        self.ratings = ratings
        self.users = sorted(users)
        self.items = sorted(items)
        self.k = k
        self.N = N
        self.C = C
        self.metric = metric

        self.user_ratings = {}
        self.user_user_sims = {}

        self.setup()

    def setup(self):
        # rating mapping
        grouped = self.ratings[['user', 'item', 'rating']].groupby('user')
        for user in self.users:
            vals = grouped.get_group(user)[['item', 'rating']].values
            self.user_ratings[user] = dict(zip(vals[:, 0].astype(int),
                                               vals[:, 1].astype(float)))

        # user similarities
        # TODO: Also let define the minimum number of co-rated items to become a relevant neighbor
        user_pairs = combinations(sorted(self.users), 2)
        for pair in user_pairs:
            self.user_user_sims[pair] = get_entity_sim(pair[0],
                                                       pair[1],
                                                       self.user_ratings,
                                                       mode=self.metric)

    # TODO: Instead of List of Tuples return List of Tuple with int user and Dict with sim and count
    def get_k_nearest_neighbors(self, user: int) -> List[Tuple[int, float]]:
        neighbors = set(self.users)
        neighbors.remove(user)

        nearest_neighbors = dict()
        for neighbor in neighbors:
            sim = self.user_user_sims[tuple(sorted((user, neighbor)))][0]
            if pd.notnull(sim):
                nearest_neighbors[neighbor] = sim

        nearest_neighbors = sorted(nearest_neighbors.items(),
                                   key=lambda kv: kv[1],
                                   reverse=True)

        return nearest_neighbors[:self.k]

    def get_neighborhood_ratings(self, user, user_neighbors: List[Tuple[int, float]]) -> Dict[int, List[Dict[str, float]]]:
        neighborhood_ratings = {}
        for neighbor, sim in user_neighbors:
            neighbor_ratings = self.user_ratings[neighbor].copy()

            # collect neighbor ratings and items
            for item, rating in neighbor_ratings.items():
                add_item = {'sim': sim, 'rating': rating}
                if item not in neighborhood_ratings.keys():
                    neighborhood_ratings[item] = [add_item]
                else:
                    neighborhood_ratings[item].append(add_item)

        # remove known items
        known_items = list(self.user_ratings[user].keys())
        for known_item in known_items:
            neighborhood_ratings.pop(known_item, None)

        return neighborhood_ratings

    # TODO: Assumes that neighborhood_ratings is nonempty, provide default answer
    @staticmethod
    def compute_rating_pred(neighborhood_ratings: dict) -> dict:
        rating_preds = dict()
        for item, ratings in neighborhood_ratings.items():
            if len(ratings) > 0:
                sims = np.array([rating['sim'] for rating in ratings])
                ratings = np.array([rating['rating'] for rating in ratings])
                pred_rating = (sims * ratings).sum() / sims.sum()
                count = len(sims)
                rating_preds[item] = {'pred': pred_rating,
                                      'count': count}
            else:
                rating_preds[item] = {'pred': None, 'count': 0}

        return rating_preds

    def compute_top_n(self, rating_preds: dict) -> OrderedDict:
        rating_preds = {key: val for (key, val) in rating_preds.items()
                        if val['count'] >= self.C}
        # try to break ties of `pred` by descending `count`
        # assuming more ratings mean higher confidence in the prediction
        sorted_rating_preds = sorted(rating_preds.items(),
                                     key=lambda kv: (kv[1]['pred'], kv[1]['count']),
                                     reverse=True)

        return OrderedDict(sorted_rating_preds[:self.N])

    # TODO: Better return List of Dicts to stay consistent with previous
    def get_recommendations(self, user: int) -> OrderedDict:
        """
        returns
        """
        user_neighbors = self.get_k_nearest_neighbors(user)
        neighborhood_ratings = self.get_neighborhood_ratings(user, user_neighbors)
        rating_preds = self.compute_rating_pred(neighborhood_ratings)
        recs = self.compute_top_n(rating_preds)

        return recs

    def get_prediction(self, user: int, item: int) -> Dict[int, Dict[str, float]]:
        """
        returns {item: {prediction, count}} for user-item combination
        """
        neighbors = self.get_k_nearest_neighbors(user)
        neighborhood_ratings = {item: list()}
        for neighbor, sim in neighbors:
            if item in self.user_ratings[neighbor].keys():
                rating = self.user_ratings[neighbor][item]
                add_item = {'sim': sim, 'rating': rating}
                neighborhood_ratings[item].append(add_item)

        pred = self.compute_rating_pred(neighborhood_ratings)

        return pred

class PopularityRecommender(object):
    """
    Popularity Recommender
    """
    def __init__(self,
                 ratings: pd.DataFrame,
                 users: np.array,
                 items: np.array,
                 N: int):
        self.ratings = ratings
        self.users = sorted(users)
        self.items = sorted(items)
        self.N = N

        self.item_popularity = None
        self.item_order = None

        self._setup()

    def _setup(self):
        self.item_popularity = self.ratings['item'].value_counts()
        self.item_order = self.item_popularity.index.values

    def get_recommendations(self, user: int) -> Dict[int, None]:
        recs = dict(zip(self.item_order[:self.N], [None]*self.N))

        return recs
