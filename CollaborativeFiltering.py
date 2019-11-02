import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.sparse import coo_matrix

class CoordDescentCollabFilter():
    def __init__(self, user_id_vec,
                 movie_id_vec,
                 rating_vec,
                 K=5,
                 learning_rate=100,
                 lmbd=0.02,
                 test=None):

        self.ratings = rating_vec
        self.userIds = user_id_vec - 1
        self.itemIds = movie_id_vec - 1

        self.U = user_id_vec.max()
        self.I = movie_id_vec.max()
        self.K = K
        self.R = coo_matrix((self.ratings, (self.userIds, self.itemIds)), shape=(self.U, self.I))

        self.P = np.random.rand(self.U, self.K).astype(np.float32)
        self.Q = np.random.rand(self.I, self.K).astype(np.float32)
        self.b_u = np.zeros(self.U)
        self.b_i = np.zeros(self.I)
        self.mu = np.mean(self.ratings[self.ratings != 0])

        self.learning_rate = learning_rate
        self.lmbd = lmbd

        self.test = test

    def train(self, epochs=5):
        for ep in range(epochs):
            P_tau = self.P[self.userIds, :]
            Q_tau = self.Q[self.itemIds, :]
            PQ_tau = (P_tau * Q_tau).sum(axis=1) + self.mu + self.b_u[self.userIds] + self.b_i[self.itemIds]
            R_hat = coo_matrix((PQ_tau, (self.userIds, self.itemIds)), shape=(self.U, self.I))
            R_hat_err = R_hat - self.R
            self.b_u = self.b_u + self.learning_rate * (
                        (self.lmbd * self.b_u - np.squeeze(np.asarray(R_hat_err.sum(axis=1)))) / R_hat.nnz)
            self.b_i = self.b_i + self.learning_rate * (
                        (self.lmbd * self.b_i - np.squeeze(np.asarray(R_hat_err.sum(axis=0)))) / R_hat.nnz)
            self.P = self.P + self.learning_rate * ((self.lmbd * self.P - R_hat_err @ self.Q) / R_hat.nnz)
            self.Q = self.Q + self.learning_rate * ((self.lmbd * self.Q - R_hat_err.T @ self.P) / R_hat.nnz)
            if (self.test is not None):
                print(f'epoch: {ep} loss: {mean_squared_error(self.test["rating"], self.predict(self.test["userId"] - 1, self.test["movieId"] - 1))}')

    def predict(self, u, i):
        P_tau = self.P[u, :]
        Q_tau = self.Q[i, :]
        PQ_tau = (P_tau * Q_tau).sum(axis=1)
        return PQ_tau + self.mu + self.b_u[u] + self.b_i[i]