import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class softmaxClassifier:
    # Q3.4 - Softmax for multi-class classification
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        10
        self.maxEvals = maxEvals
    def funObj(self, w, X, y):
        n, d = X.shape
        k = self.n_classes
        W = np.reshape(w, (k,d))
        y_binary = np.zeros((n, k)).astype(bool)
        y_binary[np.arange(n), y] = 1
        XW = np.dot(X, W.T)
        Z = np.sum(np.exp(XW), axis=1)
        # Calculate the function value
        f = - np.sum(XW[y_binary] - np.log(Z))
        # Calculate the gradient value
        g = (np.exp(XW) / Z[:,None] - y_binary).T@X
        return f, g.flatten()
    def fit(self,X, y):
        n, d = X.shape
        k = np.unique(y).size
        self.n_classes = k
        self.W = np.zeros(d*k)
        self.w = self.W # because the gradient checker is implemented in a silly way
        # Initial guess
        # utils.check_gradient(self, X, y)
        (self.W, f) = findMin.findMin(self.funObj, self.W,
        self.maxEvals, X, y, verbose=self.verbose)
        self.W = np.reshape(self.W, (k,d))
    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)
