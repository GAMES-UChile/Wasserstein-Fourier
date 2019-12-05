# collection of function to manipulate / preprocess data
#import pandas as pd
import numpy as np
import math


def load_data(name):

    if name == 'housing':
        df = pd.read_csv('../data/housing.csv', sep='\s+')
        X = df.values
        y = X[:, -1]; X = X[:, :-1]
        return X,y

    print("No dataset named {}".format(name))
    return -1


def random_split(X, y, prop_train, seed):
    """
    Given a dataset and the size of a training set return
    a random split
    """
    N = len(y)
    np.random.seed(seed)
    idx = np.arange(N)
    np.random.shuffle(idx)
    idx_tr = idx[:math.ceil(N*prop_train)]
    idx_te = idx[math.ceil(N*prop_train):]
    return X[idx_tr, :], X[idx_te, :], y[idx_tr].reshape(len(idx_tr),1), y[idx_te].reshape(len(idx_te),1)


def normalize(X_tr, X_te):
    """
    Remove training mean and standard deviation from all datasets
    """
    # Mean to 0
    m = np.mean(X_tr, 0)
    X_tr_n = X_tr - m
    X_te_n = X_te - m
    
    # X std -> 1
    std = np.std(X_tr, 0)
    X_tr_n = X_tr_n / std
    X_te_n = X_te_n / std

    return X_tr_n, X_te_n

if __name__ == '__main__':
    X, y = load_data('housing')
    X_tr, X_te, y_tr, y_te = random_split(X, y, 0.8, 123)
    normalize(X_tr, X_te)
