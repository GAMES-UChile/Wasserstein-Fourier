import numpy as np
import ot
import scipy.sparse
from scipy.spatial.distance import euclidean
from exact_barycenter import *

class WSoftmax:


    def __init__(self, X, y, support, distance='euclidean', options=None, quadratic=False, wass_square=False, robust=False):

        self.robust = robust
        self.wass_square = wass_square
        self.X = X
        self.y = y
        self.distance = distance
        self.n_class = len(np.unique(y))
        self.N = len(y)

        self.quadratic = quadratic
        if self.quadratic:
            self.weights = 0.1*np.ones((2*self.n_class+1,self.n_class))
        else:
            self.weights = 0.1*np.ones((self.n_class+1, self.n_class))

        self.support = support

        if distance == 'quantile':
            self.q_support = options['q_support']
            self.q_bars = np.zeros((self.n_class, self.q_support))

        self.bars = self.getBarycenters()
        self.W2b = self.getDistances()


    def softmax(self, z):
        if self.robust:
            z -= np.max(z)
        sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
        return sm


    def f_thetas(self, weights):
        #weights = weights.reshape(self.weights.shape)
        return np.dot(self.W2b, weights)


    def oneHot(self, y):
        m = y.shape[0]
        OHX = scipy.sparse.csr_matrix((np.ones(m), (y.astype(int), np.array(range(m)))))
        OHX = np.array(OHX.todense()).T
        return OHX


    def logp(self, weights):
        weights = weights.reshape(self.weights.shape)
        y_mask = self.oneHot(self.y)
        ft = self.f_thetas(weights)
        prob = self.softmax(ft)
        return (-1/self.N) * np.sum(y_mask * np.log(prob))


    def dlogp(self, weights):
        weights = weights.reshape(self.weights.shape)
        y_mask = self.oneHot(self.y)
        ft = self.f_thetas(weights)
        prob = self.softmax(ft)
        return (-1/self.N) * np.dot(self.W2b.T, (y_mask - prob)).reshape(-1,)


    def pred(self, x, weights=None):
        weights = weights.reshape(self.weights.shape)
        xx = self.getDistances(x)
        probs = self.softmax(np.dot(xx, weights))
        preds = np.argmax(probs, axis=1)
        return probs, preds


    def getBarycenters(self):
        bars = np.zeros((self.n_class, len(self.support)))

        for d in range(self.n_class):
            ids = self.y == d
            B = self.X[ids, :]

            if self.distance == 'euclidean' or self.distance == 'KL':
                bars[d ,:] = np.mean(B, axis=0)
            elif self.distance == 'quantile':
                bars[d,:], _, self.q_bars[d,:] = get_barycenter(B, self.support,
                                                                self.q_support)
        return bars


    def norm(self, u):
        return u / np.sum(u)


    def getDistances(self, X=None):

        if X is None:
            X = self.X

        N = len(X)
        W2b = np.ones((N, self.n_class+1)) # + 1 for bias

        if self.distance == 'euclidean':
            for i in range(N):
                for d in range(self.n_class):
                    W2b[i, d+1] = euclidean(X[i,:], self.bars[d,:])

        elif self.distance == 'quantile':
            for i in range(N):
                # TODO: interpolation method='' should be a parameter in options
                _, q = inverse_histograms(X[i,:], self.support, 
                       np.linspace(0,1,self.q_support), method='linear')
                for d in range(self.n_class):
                    W2b[i, d+1] = np.sqrt(np.sum((q - self.q_bars[d,:])**2))

        elif self.distance == 'KL':
            for i in range(N):
                for d in range(self.n_class):
                    W2b[i, d+1] = np.sum(X[i,:] * np.log(self.bars[d,:] / X[i,:]))
        else:
            print("Unknown distance resquested")
            W2b = None

        return W2b
