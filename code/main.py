import os
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.linalg import norm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from scipy.ndimage import rotate

from linear_model import softmaxClassifier
from neural_net import NeuralNet, NeuralNetSGD
from knn import KNN
# from manifold import MDS, ISOMAP
import utils
import time

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    X, y = train_set
    Xtest, ytest = test_set

    # centering matrix
    mean_point = X.mean(axis=0)
    mean_point_test = Xtest.mean(axis=0)
    X -= mean_point
    Xtest -= mean_point_test

    # normalizing training and testing data
    # https://towardsdatascience.com/going-beyond-98-mnist-handwritten-digits-recognition-cfff96337392
    mean_px = X.mean().astype(np.float32)
    mean_px_test = Xtest.mean().astype(np.float32)
    std_px = X.std().astype(np.float32)
    std_px_test = Xtest.std().astype(np.float32)
    X = (X - mean_px)/(std_px)
    Xtest = (Xtest - mean_px_test)/(std_px_test)

    # k = 81 enough to cover ~89% of variance, also to remove some noise
    # also chose this number for easier reshaping later
    k = 81
    pca = PCA(k)
    pca.fit(X)
    print("Explained variance ratio: {:.4f}".format(pca.explained_variance_ratio_.cumsum()[-1]))
    X = pca.transform(X)
    Xtest = pca.transform(Xtest)

    # randomly select 20% to rotate anywhere between -10deg to 10deg
    r = np.random.randint(5, size=np.shape(X)[0])
    X = np.reshape(X, (-1 ,9,9))
    rotate(X[r[r == 0]], random.randint(-20, 20))
    X = np.reshape(X, (-1, 81))


    if question == "mlp":
        # NN with 100 hidden layers, L1 reg, L2 loss.
        # L1 reg seems to reduce error by 0.8-1% so that's something
        # SGD doesn't work very well as we're looking for accuracy

        t = time.time()

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        hidden_layer_sizes = [100]
        model = NeuralNet(hidden_layer_sizes)
        model.fit(X,Y)
        yhat = model.predict(Xtest)
        testErr = np.mean(ytest != yhat)

        print("Test error = {:.4f}".format(testErr))
        print('Time taken: ', time.time() - t)
    
    elif question == "knn":
        # mostly hyperparameter tuning, 10 seems to be the sweet spot

        t = time.time()

        model = KNN(3)
        model.fit(X, y)
        yhat = model.predict(Xtest)
        testErr = np.mean(ytest != yhat)

        print('Test error = {:.4f}'.format(testErr))
        print('Time taken: ', time.time() - t)
    
    elif question == "reg":
        t = time.time()
        model = softmaxClassifier(maxEvals=100)
        # r = np.random.randint(5, size=np.shape(X)[0])
        # Xt = X[r == 0]
        # yt = y[r == 0]
        # model.fit(Xt,yt)
        model.fit(X,y)
        yhat = model.predict(Xtest)
        print(Xtest)
        testErr = np.mean(ytest != yhat)
        print("yhat: ", yhat)

        print('Test error = {:.4f}'.format(testErr))
        print('Time taken: ', time.time() - t)