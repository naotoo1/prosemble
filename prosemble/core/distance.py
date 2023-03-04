import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def squared_euclidean_distance(x, y):
    return euclidean_distance(x, y) ** 2
