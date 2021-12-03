import numpy as np


class LinearRegression:

    def __init__(self):
        self.learning_rate = 0.001
        self.epoch = 100000

    def y_hat(self, X, W):
        return np.dot(W.T, X)

    def cost(self, yhat, y):
        C = 1 / self.m * np.sum(np.power(yhat - y, 2))
        return C

    def gradient_descent(self, X, W, Y, yhat):
        D = 2 / self.m * np.dot(X, (yhat - y).T)
        W = W - self.learning_rate * D
    return W

    def run(self, X, Y):
        ones = np.ones((1, X.shape[1]))
        X = np.append(ones, X, axis=0)
        self.m = X.shape[1]
        self.n = X.shape[0]
        W = np.zeros((self.n, 1))

        for _ in range(self.epoch + 1):
            yhat = self.y_hat(X, W)
            cost = self.cost(yhat, y)
            w = self.gradient_descent(W, X, Y, yhat)
        return W
