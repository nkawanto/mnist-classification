import numpy as np
import findMin
import utils

# helper functions to transform between one big vector of weights
# and a list of layer parameters of the form (W,b)
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights,())])

def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes)-1):
        W_size = layer_sizes[i+1] * layer_sizes[i]
        b_size = layer_sizes[i+1]

        W = np.reshape(weights_flat[counter:counter+W_size], (layer_sizes[i+1], layer_sizes[i]))
        counter += W_size

        b = weights_flat[counter:counter+b_size][None]
        counter += b_size

        weights.append((W,b))
    return weights

def log_sum_exp(Z):
    Z_max = np.max(Z,axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:,None]), axis=1)) # per-colmumn max

class NeuralNet():
    # uses sigmoid nonlinearity
    def __init__(self, hidden_layer_sizes, lammy=1, max_iter=100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lammy = lammy
        self.max_iter = max_iter

    def funObj(self, weights_flat, X, y):
        weights = unflatten_weights(weights_flat, self.layer_sizes)

        activations = [X]
        for W, b in weights:
            Z = X @ W.T + b
            X = 1/(1+np.exp(-Z))
            activations.append(X)
        yhat = Z

        if self.classification: # softmax
            tmp = np.sum(np.exp(yhat), axis=1)
            f = -np.sum(yhat[y.astype(bool)] - log_sum_exp(yhat))
            grad = np.exp(yhat) / tmp[:,None] - y
        else:  # L2 loss
            f = 0.5*np.sum((yhat-y)**2)
            grad = yhat-y # gradient for L2 loss

        grad_W = grad.T @ activations[-2]
        grad_b = np.sum(grad, axis=0)

        g = [(grad_W, grad_b)]

        for i in range(len(self.layer_sizes)-2,0,-1):
            W, b = weights[i]
            grad = grad @ W
            grad = grad * (activations[i] * (1-activations[i])) # gradient of logistic loss
            grad_W = grad.T @ activations[i-1]
            grad_b = np.sum(grad,axis=0)

            g = [(grad_W, grad_b)] + g # insert to start of list

        g = flatten_weights(g)

        # add L1 regularization
        f += 0.5 * self.lammy * np.sum(weights_flat)
        g += self.lammy * weights_flat

        return f, g


    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:,None]

        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1]>1 # assume it's classification iff y has more than 1 column

        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes)-1):
            W = scale * np.random.randn(self.layer_sizes[i+1],self.layer_sizes[i])
            b = scale * np.random.randn(1,self.layer_sizes[i+1])
            weights.append((W,b))
        weights_flat = flatten_weights(weights)

        # utils.check_gradient(self, X, y, len(weights_flat), epsilon=1e-6)
        weights_flat_new, f = findMin.findMin(self.funObj, weights_flat, self.max_iter, X, y, verbose=True)

        self.weights = unflatten_weights(weights_flat_new, self.layer_sizes)

    def predict(self, X):
        for W, b in self.weights:
            Z = X @ W.T + b
            X = 1/(1+np.exp(-Z))
        if self.classification:
            return np.argmax(Z,axis=1)
        else:
            return Z


class NeuralNetSGD(NeuralNet):
    # uses sigmoid nonlinearity
    def __init__(self, hidden_layer_sizes, lammy=1, alpha = 0.001, epochs = 10):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lammy = lammy
        self.alpha = alpha
        self.epochs = epochs

    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:,None]

        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1]>1 # assume it's classification iff y has more than 1 column

        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes)-1):
            W = scale * np.random.randn(self.layer_sizes[i+1],self.layer_sizes[i])
            b = scale * np.random.randn(1,self.layer_sizes[i+1])
            weights.append((W,b))
        weights_flat = flatten_weights(weights)

        # utils.check_gradient(self, X, y, len(weights_flat), epsilon=1e-6)
        weights_flat_new, f = self.SGD(self.funObj, weights_flat, 500, X, y, verbose=True)

        self.weights = unflatten_weights(weights_flat_new, self.layer_sizes)

    def SGD(self, funObj, w, miniBatchSize, *args, verbose=True):
        f, g = funObj(w,*args)

        # Unpack args
        X = args[0]
        Y = args[1]
        # Calculate batches and subsets
        numMiniBatches = len(X) // miniBatchSize
        Xbatches = np.split(X, numMiniBatches)
        Ybatches = np.split(Y, numMiniBatches)

        for i in range(self.epochs):

            # Shuffle batch order
            indexOrder = np.arange(numMiniBatches)
            np.random.shuffle(indexOrder)

            # Iterate through minibatches
            for ii in range(numMiniBatches):
                # Index minibatches
                XBatch = Xbatches[indexOrder[ii]]
                YBatch = Ybatches[indexOrder[ii]]

                # Calculate  w and f
                w_new = w - self.alpha * g
                f_new, g_new = funObj(w_new, XBatch, YBatch)
                # Update
                w = w_new
                f = f_new
                g = g_new

            # Print progress for latet minibatch only (approximation)
            if verbose > 0:
                print("%d - loss: %.3f" % (i, f_new))

        return w, f