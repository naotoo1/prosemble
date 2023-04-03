"""
Implementation of npc1
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT


from sklearn.metrics import accuracy_score

import numpy as np

from .kmeans import kmeans_plusplus

from prosemble.core.distance import euclidean_distance


def softmin_funct(x):
    """

    :param x: input vector
    :return: softmin function
    """
    return np.exp(-x) / np.exp(-x).sum()


class NPC1(kmeans_plusplus):
    def __init__(self, data, c, num_inter, epsilon, ord, labels, set_prototypes_=None, opt_metric=0.8,
                 max_opt_steps=10):
        super().__init__(data, c, num_inter, epsilon, ord)
        self.labels = labels
        self.prototypes_ = self.get_prototypes_class(self.labels)[0]
        self.initial_proto = []
        self.fit_prototypes = []
        self.opt_metric = opt_metric
        self.max_opt_steps = max_opt_steps
        self.set_prototypes_ = set_prototypes_

        if not isinstance(opt_metric, float):
            raise TypeError('opt_metric must be a float')
        if not isinstance(max_opt_steps, int):
            raise ValueError('max_opt_metric')

    def fit_optim(self):
        train, prototypes, counter = True, self.prototypes_, 0
        self.initial_proto.append(prototypes)
        while train:
            counter += 1
            pred = [self._nearest_centroids(sample=sample, centroids=prototypes) for sample in self.data]
            get_accuracy = accuracy_score(self.labels, pred)
            if get_accuracy >= self.opt_metric or counter == self.max_opt_steps:
                self.fit_prototypes.append(prototypes)
                train = False
            prototypes = self.get_prototypes_class(self.labels)[0]
        return None

    def fit(self):
        if self.set_prototypes_ is None:
            return self.fit_optim()
        return self.fit_prototypes.append(self.set_prototypes_)

    def predict_sample(self, x):
        try:
            return [self._nearest_centroids(sample, self.fit_prototypes[0]) for sample in x]
        except IndexError:
            print('fit the model before calling predict')

    def prototypes(self):
        return self.fit_prototypes[0]

    def initial_prototypes_(self):
        if self.set_prototypes_ is None:
            return self.initial_proto[0]
        return self.fit_prototypes[0]

    def get_distance_(self, x):
        return np.array(
            [[euclidean_distance(sample, centroid)
              for centroid in self.prototypes()] for index, sample in enumerate(x)]
        )

    def get_predict_proba(self, x):
        distance_space = self.get_distance_(x)
        return np.array([softmin_funct(v) for _, v in enumerate(distance_space)])
