import numpy as np
import ot
import scipy.sparse
from scipy.spatial.distance import euclidean
from exact_barycenter import *

class SoftmaxALL:


    def __init__(self, X, y, support, options=None, robust=False):

        self.robust = robust
        self.X = X
        self.y = y
        self.n_class = len(np.unique(y))
        self.N = len(y)

        self.weights = 0.1*np.ones((3*self.n_class+1, self.n_class))

        self.support = support

        self.q_support = options['q_support']
        self.q_bars = np.zeros((self.n_class, self.q_support))

        self.wbars = self.getBarycenters('quantile')
        self.l2bars = self.getBarycenters('euclidean')

        self.features = self.get_features()


    def get_features(self, X=None):
        d2wb = self.getDistances('quantile', X)
        d2l2b = self.getDistances('euclidean', X)
        d2klb = self.getDistances('KL', X)
        N = len(d2klb)
        features = np.concatenate((d2wb, d2klb, d2l2b, np.ones((N,1))), axis=1)
        return features

    def softmax(self, z):
        if self.robust:
            z -= np.max(z)
        sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
        return sm


    def f_thetas(self, weights):
        #weights = weights.reshape(self.weights.shape)
        return np.dot(self.features, weights)


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
        return (-1/self.N) * np.dot(self.features.T, (y_mask - prob)).reshape(-1,)


    def pred(self, x, weights=None):
        weights = weights.reshape(self.weights.shape)
        xx = self.get_features(x)
        probs = self.softmax(np.dot(xx, weights))
        preds = np.argmax(probs, axis=1)
        return probs, preds


    def getBarycenters(self, distance):
        bars = np.zeros((self.n_class, len(self.support)))

        for d in range(self.n_class):
            ids = self.y == d
            B = self.X[ids, :]

            if distance == 'euclidean' or distance == 'KL':
                bars[d ,:] = np.mean(B, axis=0)
            elif distance == 'quantile':
                bars[d,:], _, self.q_bars[d,:], _ = get_barycenter(B, self.support,
                                                                self.q_support)
        return bars


    def norm(self, u):
        return u / np.sum(u)


    def getDistances(self, distance, X=None):

        if X is None:
            X = self.X

        N = len(X)
        d2b = np.ones((N, self.n_class))

        if distance == 'euclidean':
            for i in range(N):
                for d in range(self.n_class):
                    d2b[i, d] = euclidean(X[i,:], self.l2bars[d,:])

        elif distance == 'quantile':
            for i in range(N):
                # TODO: interpolation method='' should be a parameter in options
                _, q = inverse_histograms(X[i,:], self.support, 
                       np.linspace(0,1,self.q_support), method='linear')
                for d in range(self.n_class):
                    d2b[i, d] = np.sqrt(np.sum((q - self.q_bars[d,:])**2))

        elif distance == 'KL':
            for i in range(N):
                for d in range(self.n_class):
                    d2b[i, d] = np.sum(X[i,:] * np.log(X[i,:] / self.l2bars[d,:]))
        else:
            print("Unknown distance resquested")
            d2b = None

        return d2b
