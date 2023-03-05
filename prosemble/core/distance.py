"""
Prosemble distances
"""

import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def squared_euclidean_distance(x, y):
    return euclidean_distance(x, y) ** 2


def manhattan_distance(point1, point2):
    sum = 0
    for x1, x2 in zip(point1, point2):
        difference = x2 - x1
        absolute_difference = abs(difference)
        sum += absolute_difference
    return sum
