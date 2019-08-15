import csv
from scipy.spatial.distance import euclidean
import random
import math
import operator
import numpy as np
from misc import *
from exact_barycenter import *


name2int = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}


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

#def euclideanDistance(instance1, instance2, length):
#    distance = 0
#    for x in range(length):
#        distance += pow((instance1[x] - instance2[x]), 2)
#    return math.sqrt(distance)


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
    ks = [3, 4, 5, 7, 10, 15, 20]
    methods = ['euclidean', 'quantile']
    n_seeds = 10
    results = np.zeros((len(methods), len(ks), n_seeds))
    #X_tr, X_te, y_tr, y_te = loadDataset('/Users/ard/Downloads/iris.data', split)
    X = np.load('X.npy') 
    y = np.load('y.npy')

    # generate predictions
    predictions=[]
    for s in range(n_seeds):
        X_tr, X_te, y_tr, y_te = random_split(X, y, prop_train=split, seed=s) 
        y_tr = y_tr.reshape((len(y_tr),))
        y_te = y_te.reshape((len(y_te),))

        for idx_m, m in enumerate(methods):
            for idx_k, k in enumerate(ks):
                predictions = []
                for idx, x_te in enumerate(X_te):

                    idx_neighbors = getNeighbors(X_tr, x_te, k, m)
                    result = getResponse(y_tr[idx_neighbors])
                    predictions.append(result)

                results[idx_m, idx_k, s] = getAccuracy(y_te, predictions)
                print('Accuracy: ' + str(results[idx_m, idx_k, s])  + '%, method ' + m + ', k = '+ str(k))
    np.save('results_knn', results)
main()
