"""
Implementation of KNN Algorithm
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT


import numpy as np
from collections import Counter

from prosemble.core.distance import euclidean_distance

# def euclidean_distance(x, y):
#     return np.sqrt(np.sum((x - y) ** 2))


def get_mode(x):
    counter = Counter(x)
    return [k for k, v in counter.items() if v == max(counter.values())][0], \
        max([value / sum(counter.values()) for value in counter.values()])


def get_distance_space(dataset, test_data):
    return [euclidean_distance(test_data, sample) for index, sample in enumerate(dataset)]


class KNN:
    def __init__(self, dataset, labels, c):
        self.data = dataset
        self.labels = labels
        self.neighbours = c
        self.nearest = []
        self.num_classes = len(np.unique(np.array(labels)))

    def predict_(self, test_data):

        distance_space = get_distance_space(self.data, test_data)
        indices_min_distance_space = np.argsort(distance_space)
        k_nearest_indices = indices_min_distance_space[:self.neighbours]
        k_nearest_neighbour_labels = [
            self.labels[k_index] for _, k_index in enumerate(k_nearest_indices)
        ]
        return get_mode(k_nearest_neighbour_labels)

    def predict(self, test_data):
        """

        :param test_data: input vector
        :return: label(s) og the input vector
        """
        if len(test_data) == 1:
            return self.predict_(test_data)[0]
        return [self.predict_(sample)[0] for sample in test_data]

    def get_proba(self, test_data):
        """

        :param test_data: input vector
        :return: confidence of the predicted input vector
        """
        if len(test_data) == 1:
            return self.predict_(test_data)[1]
        return [self.predict_(sample)[1] for sample in test_data]

    def distance_space(self, test_data):
        """

        :param test_data: input vector
        :return: distance space
        """
        return np.sort(get_distance_space(self.data, test_data))
