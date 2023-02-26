import numpy as np


def get_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def get_distance_squared(x, y):
    return get_distance(x, y) ** 2
