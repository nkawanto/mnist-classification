import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

def log_sum_exp(Z):
    Z_max = np.max(Z,axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:,None]), axis=1)) # per-colmumn max

class SVM():
    # L2 Regularized Logistic Regression (no intercept)
    # haha SIKE it's now SVM(?)
    def __init__(self, lammy=1.0, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):

        n, d = X.shape
        self.num_classes = np.unique(y).size

        W = np.reshape(w, (self.num_classes, d))

        XW = np.dot(X, W.T)

        oneHotEncoding = np.zeros((n, self.num_classes))
        oneHotEncoding[np.arange(n),y] = 1
        oneHotEncoding = oneHotEncoding.astype(np.bool_)

        XW = XW[oneHotEncoding]

        secondTerm = log_sum_exp(np.dot(X, W.T))
        f = 0
        # Calculate the function value
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if (j != i):
                    f += 0.5*self.lammy*np.sum(w**2) + np.sum(max(0,1 - np.sum(y[y==i]*XW[i]) + np.sum(y[y==j]*XW[j])))

        # Calculate the gradient value
        dotProduct = np.dot(X, W.T)

        g = (np.exp(dotProduct) / np.expand_dims(np.sum(np.exp(dotProduct), axis=1), 1))
        g = g - oneHotEncoding
        g = np.dot(g.T, X)
        g = np.reshape(g, (self.num_classes * d))
        return f, g.flatten()

    def fit(self,X, y):
        n, d = X.shape
        k = np.unique(y).size
        self.n_classes = k
        self.W = np.zeros(d*k)
        self.w = self.W # because the gradient checker is implemented in a silly way
        # Initial guess
        (self.W, f) = findMin.findMin(self.funObj, self.W, self.maxEvals, X, y, verbose=self.verbose)
        self.W = np.reshape(self.W, (k,d))

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)



