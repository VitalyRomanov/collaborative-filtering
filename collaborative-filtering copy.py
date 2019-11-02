#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


# In[2]:


ratings_path = "/Volumes/External/datasets/Recommendation/MovieLens/ml-20m/ratings.csv"
titles_path = "/Volumes/External/datasets/Recommendation/MovieLens/ml-20m/movies.csv"

titles = pd.read_csv(titles_path).drop('genres', axis=1)
data = pd.read_csv(ratings_path).drop('timestamp', axis=1)
data_clean = data.groupby('movieId').filter(lambda x: len(x) >= 20).groupby('userId').filter(lambda x: len(x) >= 20)

del data


# In[3]:


n_users = data_clean['userId'].nunique()
n_movies = data_clean['movieId'].nunique()
tr_ratings = np.zeros((n_users, n_movies))
te_ratings = np.zeros((n_users, n_movies))


# In[4]:


split_mask = np.random.rand(len(data_clean)) < 0.8
train = data_clean[split_mask]
test = data_clean[~split_mask]
tr_n_pairs = train.shape[0]
te_n_pairs = test.shape[0]


# In[5]:


new_id, old_id = zip(*enumerate(data_clean['movieId'].unique()))
old_m2new = dict(zip(old_id, new_id))
new_id, old_id = zip(*enumerate(data_clean['userId'].unique()))
old_u2new = dict(zip(old_id, new_id))

del data_clean


# In[89]:


def create_mask(ratings):
    mask = np.zeros_like(ratings)
    mask[np.where(ratings!=0.)] = 1.
    return mask

def construct_rating_matr(data, n_users, n_items):
    ratings = np.zeros((n_users, n_movies))
    np_data = data.values
    userId = np.vectorize(lambda x: old_u2new[x])(np_data[:,0])
    movieId = np.vectorize(lambda x: old_m2new[x])(np_data[:,1])
    ratings_val = np_data[:,2]
    ratings[userId, movieId] = ratings_val
    return ratings, create_mask(ratings)

def construct_rating_matr_sp(data, n_users, n_items):
    np_data = data.values
    userId = np.vectorize(lambda x: old_u2new[x])(np_data[:,0])
    movieId = np.vectorize(lambda x: old_m2new[x])(np_data[:,1])
    ratings_val = np_data[:,2]
    ratings = coo_matrix((ratings_val, (userId, movieId)), shape=(n_users, n_items))
    return ratings, userId, movieId

# tr_ratings, tr_mask = construct_rating_matr_sp(train)
# te_ratings, te_mask = construct_rating_matr_sp(test)
tr_ratings, tr_users, tr_movies = construct_rating_matr_sp(train, n_users, n_movies)
te_ratings, te_users, te_movies = construct_rating_matr_sp(test, n_users, n_movies)

del train, test


# In[55]:


decom_rank = 100
l = 2.
lr_decay = 1.0


# In[110]:


def estimate_ratings(users_m, movies_m):
    return users_m @ movies_m.T

def estimate_ratings_sp(users_m, movies_m, userId, movieId, shape): 
    inner = np.sum(users_m[userId] * movies_m[movieId], axis=1)
    return coo_matrix((inner, (userId, movieId)), shape=shape)

def error(pred, true):
    return pred - true 

def apply_mask(error, mask):
    return np.multiply(error, mask)

def gradient_sp(users_m, movies_m, gradients_for, l, true_r, total_ratings, userId, movieId):
    pred_r = estimate_ratings_sp(users_m, movies_m, userId, movieId, true_r.shape)
    err = error(pred_r, true_r)
    
    if gradients_for=='u':
        grad = err @ movies_m + l * users_m
    elif gradients_for=='m':
        grad = err.T @ users_m + l * movies_m
        
    return grad / total_ratings

def gradient(users_m, movies_m, gradients_for, l, mask, true_r, total_ratings):
    pred_r = estimate_ratings(users_m, movies_m)
    err = error(pred_r, true_r)
    m_err = apply_mask(err, mask)
    
    if gradients_for=='u':
        grad = m_err @ movies_m + l * users_m
    elif gradients_for=='m':
        grad = m_err.T @ users_m + l * movies_m
        
    return grad / total_ratings

def rmse_sp(users_m, movies_m, l, true_r, userId, movieId, total_ratings):
    pred_r = estimate_ratings_sp(users_m, movies_m, userId, movieId, (users_m.shape[0], movies_m.shape[0]))
    err = error(pred_r, true_r)
    return np.sqrt(err.power(2).sum()/total_ratings)

def rmse(users_m, movies_m, l, mask, true_r, total_ratings):
    pred_r = estimate_ratings(users_m, movies_m)
    err = error(pred_r, true_r)
    m_err = apply_mask(err, mask)
    return np.sqrt(np.sum(np.square(m_err))/total_ratings)

def loss(users_m, movies_m, l, mask, true_r, total_ratings):
    pred_r = estimate_ratings(users_m, movies_m)
    err = error(pred_r, true_r)
    m_err = apply_mask(err, mask)
    
    return rmse(users_m, movies_m, l, mask, true_r, total_ratings) + l * (np.linalg.norm(users_m) + np.linalg.norm(movies_m))
    
# rating_estimate_masked = lambda users, movies, mask: np.multiply(np.matmul(users, movies.T), mask)
# error_est = lambda true, pred:  pred - true
# gradient = lambda error, weights_to_upd, weights_other, l: 1 / tr_n_pairs * np.matmul(error, weights_other) + l * weights_to_upd
# loss = lambda user, movies, l, ratings, mask: 1 / te_n_pairs * np.sum(np.square(ratings - rating_estimate_masked(user, movies, mask))) + \
# l * (np.linalg.norm(user_m) + np.linalg.norm(item_m))
# grad_u = lambda user, movies, l: 


# In[ ]:


# gradient(user_m, item_m, 'm', l, tr_mask, tr_ratings, 1.)


# In[ ]:


# for i in range(max_iter):
#     user_m = user_m - lr * gradient(error_est(tr_ratings, rating_estimate_masked(user_m, item_m, tr_mask)), user_m, item_m, l)
#     item_m = item_m - lr * gradient(error_est(tr_ratings, rating_estimate_masked(user_m, item_m, tr_mask)).T, item_m, user_m, l)
#     print(loss(user_m, item_m, l, te_ratings, te_mask))
#     lr *= lr_decay


# In[ ]:


## Dense
# user_m = np.random.rand(n_users, decom_rank)
# item_m = np.random.rand(n_movies, decom_rank)
# lr = 100.0
# max_iter = 20
# for i in range(max_iter):
#     user_m = user_m - lr * gradient(user_m, item_m, 'u', l, tr_mask, tr_ratings, tr_n_pairs)
#     item_m = item_m - lr * gradient(user_m, item_m, 'm', l, tr_mask, tr_ratings, tr_n_pairs)
#     print(rmse(user_m, item_m, l, te_mask, te_ratings, te_n_pairs))
#     lr *= lr_decay


# In[111]:


user_m = np.random.rand(n_users, decom_rank)
item_m = np.random.rand(n_movies, decom_rank)
lr = 100.0
max_iter = 10
for i in range(max_iter):
    user_m = user_m - lr * gradient_sp(user_m, item_m, 'u', l, tr_ratings, tr_n_pairs, tr_users, tr_movies)
    item_m = item_m - lr * gradient_sp(user_m, item_m, 'm', l, tr_ratings, tr_n_pairs, tr_users, tr_movies)
    print(rmse_sp(user_m, item_m, l, te_ratings, te_users, te_movies, te_n_pairs))
    lr *= lr_decay

