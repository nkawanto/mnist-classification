
import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        X = self.X
        y = self.y
        n = X.shape[0]
        t = Xtest.shape[0]
        k = min(self.k, n)

        dist2 = utils.euclidean_dist_squared(X, Xtest)

        yhat = np.ones(t, dtype=np.uint8)
        for i in range(t):
            inds = np.argsort(dist2[:,i])
            yhat[i] = stats.mode(y[inds[:k]])[0][0]

        return yhat