#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


class Recommender:
    def __init__(self, lambd=2.0,
                 learn_rate=200.,
                 max_iter=2000,
                 decomposition_rank=5):

        self.lambd = lambd
        self.learn_rate = learn_rate
        self.max_iter = max_iter
        self.decomposition_rank = decomposition_rank

    def fit(self, data, test_data=None):
        """
        Learn model parameters
        :param data: training data
        :param test_data: test data, if provided, used to calculate RMSE error
        """

        self._prep_for_data(data)

        self.allocate_parameters()

        # no need to train mean rating, since its value is evident
        # self.mean_rating = data[:, 2].mean()

        ratings, userId, itemId = self._construct_sparse_rating_matr(data)
        if test_data is not None:
            t_ratings, t_userId, t_itemId = \
                self._construct_sparse_rating_matr(test_data)

        for i in range(self.max_iter):
            param_grad, bias_grad = \
                self._param_gradients("users", ratings, userId, itemId)
            self.user_param -= self.learn_rate * param_grad
            # self.user_bias -= self.learn_rate * bias_grad

            param_grad, bias_grad = \
                self._param_gradients("items", ratings, userId, itemId)
            self.item_param -= self.learn_rate * param_grad
            # self.item_bias -= self.learn_rate * bias_grad

            if test_data is not None:
                print("Epoch %d, test rmse %.4f" %
                      (i, self._rmse(t_ratings, t_userId, t_itemId)))
            else:
                print("Epoch %d, train rmse %.4f" %
                      (i, self._rmse(ratings, userId, itemId)))

    def _rmse(self, ratings, userId, itemId):
        """
        Estimate RMSE on arbitrary data
        :param ratings: sparse coo_matrix with ratings
        :param userId: compacted user ids
        :param itemId: compacted item ids
        :return: rmse value
        """
        n_ratings = len(userId)

        ratings_pred = self._estimate_ratings_sparse(userId, itemId)
        error = ratings_pred - ratings

        return np.sqrt(error.power(2).sum() / n_ratings)

    def _param_gradients(self, gradients_for, ratings, userId, itemId):
        """
        Compute gradients for weights and biases
        :param gradients_for: str that takes value 'users' or 'items'
        :param ratings: sparse matrix with ratings
        :param userId: compacted user ids
        :param itemId: compacted item ids
        :return: parameter gradients, bias gradients
        """

        ratings_pred = self._estimate_ratings_sparse(userId, itemId)
        error = ratings_pred - ratings

        if gradients_for == 'users':
            # constant factors are avoided and hidden in learning rate
            param_grad = error @ self.item_param + self.lambd * self.user_param
            # shape (n_users, decomposition_rank)
            bias_grad = error.sum(axis=1)
            # shape (n_users, 1)

            assert param_grad.shape == self.user_param.shape
            assert bias_grad.shape == self.user_bias.shape

        elif gradients_for == 'items':
            param_grad = error.T @ self.user_param + self.lambd * self.item_param
            # shape (n_items, decomposition_rank)
            bias_grad = error.sum(axis=0).reshape(-1, 1)
            # shape (n_items, 1)

            assert param_grad.shape == self.item_param.shape
            assert bias_grad.shape == self.item_bias.shape
        else:
            raise ValueError("Unsupported value for 'gradients_for': %s"
                             % gradients_for)

        # normalize gradients
        return param_grad / self.train_size, bias_grad / self.train_size

    def _estimate_ratings_sparse(self, userId, itemId):
        """
        Estimate ratings as coo_matrix using matrix factorization
        :param userId: compacted user ids
        :param itemId: compacted item ids
        :return: coo matrix with shape (n_users, n_items) with estimated ratings
        """

        u_b = self.user_bias[userId].reshape(-1, )  # shape (total_ratings, )
        i_b = self.item_bias[itemId].reshape(-1, )  # shape (total_ratings, )

        inner_prod = (self.user_param[userId] * self.item_param[itemId]).sum(axis=1)
        # shape (total_ratings, 1)

        assert inner_prod.shape == u_b.shape
        assert inner_prod.shape == i_b.shape

        rating = inner_prod + self.mean_rating + u_b + i_b
        return coo_matrix((rating, (userId, itemId)),
                          shape=(self.n_users, self.n_items))

    def _construct_sparse_rating_matr(self, data):
        """
        Constructs sparse rating matrix in coo format
        :param data: training data
        :return: ratings: coo sparse matrix with shape (n_users, n_items)
                userId: compacted user ids with shape (n_users, )
                itemId: compacted user ids with shape (n_items, )
        """
        # vectorization (rather than map) helps to avoid allocating
        # additional memory
        # map original ids to compacted ids
        # see _prep_for_data for details
        userId = np.vectorize(lambda x: self.old_u2new[x])(data[:, 0])
        # shape (total_ratings, )
        movieId = np.vectorize(lambda x: self.old_m2new[x])(data[:, 1])
        # shape (total_ratings, )
        ratings_val = data[:, 2]
        # shape (total_ratings, )

        ratings = coo_matrix((ratings_val, (userId, movieId)),
                             shape=(self.n_users, self.n_items))

        return ratings, userId, movieId

    def allocate_parameters(self):
        self.user_param = np.random.rand(self.n_users, self.decomposition_rank)
        self.item_param = np.random.rand(self.n_items, self.decomposition_rank)
        self.mean_rating = 0.0
        self.user_bias = np.zeros((self.n_users, 1))
        self.item_bias = np.zeros((self.n_items, 1))

    def _prep_for_data(self, data):
        """
        Compact user ids and item ids so that ids come from a continuous interval
        Store the total numbed of items and users as class member
        :param data: training data ndarray in the with columns 'user_id',
                    'item_id', 'rating'
        """

        def to_continuous_ids(train):
            old_ids = list(set(train))
            old_ids.sort()
            new_id, old_id = zip(*enumerate(old_ids))
            return dict(zip(old_id, new_id))

        self.old_m2new = to_continuous_ids(data[:, 1]) #dict: old_id -> new_id
        self.old_u2new = to_continuous_ids(data[:, 0])
        self.n_users = len(self.old_u2new)
        self.n_items = len(self.old_m2new)
        self.train_size = data.shape[0]


# read data
# data format: userid, itemid, rating
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

mfr = Recommender()
mfr.fit(train.values, test.values)
