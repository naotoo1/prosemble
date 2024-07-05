"""
Prosemble distances
"""

import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def squared_euclidean_distance(x, y):
    return euclidean_distance(x, y) ** 2


def manhattan_distance(point1, point2):
    sum_ = 0
    for x1, x2 in zip(point1, point2):
        difference = x2 - x1
        absolute_difference = abs(difference)
        sum_ += absolute_difference
    return sum_

def lpnorm_distance(x, y, p):
    distances = np.linalg.norm((x - y), ord=p)
    return distances

def omega_distance(x, y, omega):
    projected_x = x @ omega
    projected_y = y @ omega
    distances = squared_euclidean_distance(projected_x, projected_y)
    return distances

def lomega_distance(x, y, omegas):
    projected_x = x @ omegas
    projected_y = (np.array(y @ omegas).diagonal()).T 
    expanded_y = np.expand_dims(projected_y, axis=1)
    differences_squared = (expanded_y - projected_x)**2
    distances = np.sum(differences_squared, axis=2)
    distances = distances.transpose(1, 0)
    return distances
