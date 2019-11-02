import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix

ratings_path = "/Volumes/External/datasets/Recommendation/MovieLens/ml-20m/ratings-1m.csv"
titles_path = "/Volumes/External/datasets/Recommendation/MovieLens/ml-20m/movies.csv"

titles = pd.read_csv(titles_path).drop('genres', axis=1)
data = pd.read_csv(ratings_path).drop('timestamp', axis=1)
data_clean = data.groupby('movieId').filter(lambda x: len(x) >= 20).groupby('userId').filter(lambda x: len(x) >= 20)

n_users = data_clean['userId'].nunique()
n_movies = data_clean['movieId'].nunique()
tr_ratings = np.zeros((n_users, n_movies))
te_ratings = np.zeros((n_users, n_movies))

split_mask = np.random.rand(len(data_clean)) < 0.8
train = data_clean[split_mask]
test = data_clean[~split_mask]

new_id, old_id = zip(*enumerate(data_clean['movieId'].unique()))
old_m2new = dict(zip(old_id, new_id))
new_id, old_id = zip(*enumerate(data_clean['userId'].unique()))
old_u2new = dict(zip(old_id, new_id))

def construct_rating_matr(data, n_users, n_items):
    ratings = np.zeros((n_users, n_movies))
    np_data = data.values
    userId = np.vectorize(lambda x: old_u2new[x])(np_data[:,0])
    movieId = np.vectorize(lambda x: old_m2new[x])(np_data[:,1])
    ratings_val = np_data[:,2]
    ratings[userId, movieId] = ratings_val
    return ratings

def construct_rating_matr_sp(data, n_users, n_items):
    ratings = dok_matrix((n_users, n_items), dtype=np.float32)
    np_data = data.values
    userId = np.vectorize(lambda x: old_u2new[x])(np_data[:,0])
    movieId = np.vectorize(lambda x: old_m2new[x])(np_data[:,1])
    ratings_val = np_data[:,2]
    ratings[userId, movieId] = ratings_val
    ratings = ratings.tocsc()
    return ratings, userId, movieId

tr_ratings, tr_users, tr_movies = construct_rating_matr_sp(train, n_users, n_movies)
te_ratings, te_users, te_movies = construct_rating_matr_sp(test, n_users, n_movies)

decom_rank = 100
l = 2.
lr_decay = 1.0

def create_mask(ratings):
    mask = np.zeros_like(ratings)
    mask[np.where(ratings!=0.)] = 1.
    return mask

tr_mask = create_mask(tr_ratings)
te_mask = create_mask(te_mask)

tr_n_pairs = np.sum(tr_mask)
te_n_pairs = np.sum(te_mask)


def estimate_ratings(users_m, movies_m):
    return users_m @ movies_m.T


def estimate_ratings_sp(users_m, movies_m, userId, movieId, shape):
    inner = np.sum(users_m[userId] * movies_m[movieId], axis=1)
    ratings = dok_matrix(shape, dtype=np.float32)
    ratings[userId, movieId] = inner
    return ratings.tocsc()


def error(pred, true):
    return pred - true


def apply_mask(error, mask):
    return np.multiply(error, mask)


def gradient_sp(users_m, movies_m, gradients_for, l, true_r, total_ratings, userId, movieId):
    pred_r = estimate_ratings_sp(users_m, movies_m, userId, movieId, true_r.shape)
    err = error(pred_r, true_r)

    if gradients_for == 'u':
        grad = err @ movies_m + l * users_m
    elif gradients_for == 'm':
        grad = err.T @ users_m + l * movies_m

    return grad / total_ratings


def gradient(users_m, movies_m, gradients_for, l, mask, true_r, total_ratings):
    pred_r = estimate_ratings(users_m, movies_m)
    err = error(pred_r, true_r)
    m_err = apply_mask(err, mask)

    if gradients_for == 'u':
        grad = m_err @ movies_m + l * users_m
    elif gradients_for == 'm':
        grad = m_err.T @ users_m + l * movies_m

    return grad / total_ratings


def rmse_sp(users_m, movies_m, l, true_r, userId, movieId, total_ratings):
    pred_r = estimate_ratings_sp(users_m, movies_m, userId, movieId, (users_m.shape[0], movies_m.shape[0]))
    err = error(pred_r, true_r)
    return np.sqrt(np.sum(np.square(err)) / total_ratings)


def rmse(users_m, movies_m, l, mask, true_r, total_ratings):
    pred_r = estimate_ratings(users_m, movies_m)
    err = error(pred_r, true_r)
    m_err = apply_mask(err, mask)
    return np.sqrt(np.sum(np.square(m_err)) / total_ratings)


def loss(users_m, movies_m, l, mask, true_r, total_ratings):
    pred_r = estimate_ratings(users_m, movies_m)
    err = error(pred_r, true_r)
    m_err = apply_mask(err, mask)

    return rmse(users_m, movies_m, l, mask, true_r, total_ratings) + l * (
                np.linalg.norm(users_m) + np.linalg.norm(movies_m))


user_m = np.random.rand(n_users, decom_rank)
item_m = np.random.rand(n_movies, decom_rank)
lr = 100.0
max_iter = 20
for i in range(max_iter):
    user_m = user_m - lr * gradient_sp(user_m, item_m, 'u', l, tr_ratings, tr_n_pairs, tr_userId, tr_movieId)
    item_m = item_m - lr * gradient_sp(user_m, item_m, 'm', l, tr_ratings, tr_n_pairs, tr_userId, tr_movieId)
    print(rmse_sp(user_m, item_m, l, te_ratings, te_userId, te_movieId, te_n_pairs))
    lr *= lr_decay