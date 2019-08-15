import csv
from scipy.spatial.distance import euclidean
import random
import math
import operator
import numpy as np
from misc import *
from exact_barycenter import *


def loadDataset(filename, split):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        N = len(dataset)-1
        d = len(dataset[0])
        X = np.zeros((N,d-1))
        y = np.zeros((N,))

        for i in range(len(dataset)-1):
            for j in range(4):
                X[i][j] = float(dataset[i][j])
            y[i] = name2int[dataset[i][-1]]

        return random_split(X,y, prop_train=split, seed=13)


def getDistance(x1, x2, distance='euclidean'):

    if distance == 'euclidean':
        return euclidean(x1, x2)

    elif distance == 'quantile':
        support = np.linspace(0, 0.5, 501)
        # TODO: interpolation method='' should be a parameter in options
        _, q1 = inverse_histograms(x1, support, np.linspace(0, 1, 1000))
        _, q2 = inverse_histograms(x2, support, np.linspace(0, 1, 1000))
        return np.sqrt(np.sum((q1 - q2)**2))
    else:
        print('UNKNOWN DISTANCE')
        return -1


def getNeighbors(X_tr, x_te, k, distance):
    distances = []

    for idx, x in enumerate(X_tr):
        dist = getDistance(x, x_te, distance)
        distances.append((idx, dist))
    distances.sort(key=operator.itemgetter(1))
    idx_neighbors = []
    for x in range(k):
        idx_neighbors.append(distances[x][0])
    return idx_neighbors


def getResponse(neighbors_label):
    classVotes = {}
    for n in neighbors_label:
        if n in classVotes:
            classVotes[n] += 1
        else:
            classVotes[n] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(y_te, predictions):
    correct = 0
    for idx, y in enumerate(y_te):
        if y == predictions[idx]:
            correct += 1
    return (correct/float(len(y_te))) * 100.0


def main():
    # prepare data
    trainingSet=[]
    testSet=[]
    split = 0.80
    ks = [4, 5]
    methods = ['euclidean', 'quantile']
    n_seeds = 100
    results = np.zeros((len(methods), n_seeds))

    X = np.load('X.npy') 
    y = np.load('y.npy')

    # generate predictions
    predictions=[]
    for s in range(n_seeds):
        X_tr, X_te, y_tr, y_te = random_split(X, y, prop_train=split, seed=s)
        y_tr = y_tr.reshape((len(y_tr),))
        y_te = y_te.reshape((len(y_te),))

        for idx_m, (m, k) in enumerate(zip(methods,ks)):
                predictions = []
                for idx, x_te in enumerate(X_te):

                    idx_neighbors = getNeighbors(X_tr, x_te, k, m)
                    result = getResponse(y_tr[idx_neighbors])
                    predictions.append(result)

                results[idx_m, s] = getAccuracy(y_te, predictions)
                print('Accuracy: ' + str(results[idx_m, s])  + '%, method ' + m + ', k = '+ str(k))
        print('Summary -  euclidean : ' +str(np.mean(results[0, :s+1])))
        print('Summary -  quantile  : ' +str(np.mean(results[1, :s+1])))

    np.save('results_knn_4_5', results)
main()
