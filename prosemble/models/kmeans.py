"""
Implementation of Kmeans++ Alternating Optimisation Algorithm
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT

from .hcm import Kmeans
from prosemble.core.distance import euclidean_distance

import matplotlib

import numpy as np

# matplotlib.use('QtAgg')


def softmin_funct(x):
    """

    :param x: input vector
    :return: softmin function
    """
    return np.exp(-x) / np.exp(-x).sum()


class kmeans_plusplus:
    """
    params:

    data : array-like:
            input data

    c: int:
        number of clusters

    num_iter: int:
        number of iterations

    epsilon: float:
        small difference for termination of algorithm

    ord:  {non-zero int, inf, -inf, ‘fro’, ‘nuc’}
          order of the norm

    plot_steps: bool:
        True for visualisation of training and False otherwise

    """

    def __init__(self, data, c, num_inter, epsilon, ord, plot_steps=False):

        self.data = data
        self.num_clusters = c
        self.num_iter = num_inter
        self.epsilon = epsilon
        self.ord = ord
        self.plot_steps = plot_steps

        self.model = Kmeans(
            data=self.data,
            c=self.num_clusters,
            num_inter=self.num_iter,
            epsilon=self.epsilon,
            ord=self.ord,
            set_prototypes=self.get_prototypes_(),
            plot_steps=self.plot_steps
        )

        if not isinstance(self.plot_steps, bool):
            raise ValueError('must be a True')

    def _get_random_initial_prototype(self):
        random_selection = np.random.choice(self.data.shape[0], replace=False)
        return self.data[random_selection]

    def _get_sample_max_prob_margin(self):
        sample_index = np.argmax([
            euclidean_distance(sample, self._get_random_initial_prototype())
            for index, sample in enumerate(self.data)])
        return self.data[sample_index]

    @staticmethod
    def _nearest_centroids(sample, centroids):
        return np.argmin([euclidean_distance(sample, centroid) for centroid in centroids])

    def initialise_cluster(self, centroids):
        clusters = [[] for _ in range(len(centroids))]
        for index, sample in enumerate(self.data):
            centroid_index = self._nearest_centroids(sample, centroids)
            clusters[centroid_index].append(index)
        return clusters

    def compute_objective_function(self, clusters, centroids):
        return np.sum(
            [self._distance_(self.data[clusters[i]], centroids[i]) for i in range(len(centroids))]
        )

    @staticmethod
    def _distance_(samples, centroid):
        return np.sum(
            [euclidean_distance(sample, centroid) for index, sample in enumerate(samples)]
        )

    @staticmethod
    def not_nearest_centroids(samples, centroid):
        try:
            distance = [euclidean_distance(sample, centroid) for sample in samples]
            return [max(distance), np.argmax(distance)]
        except ValueError:
            pass
        return None

    def get_prototypes_(self):
        prototypes = [self._get_random_initial_prototype(), self._get_sample_max_prob_margin()]
        while len(prototypes) < self.num_clusters:
            clusters = self.initialise_cluster(centroids=prototypes)
            not_nearest_centroids = [
                self.not_nearest_centroids(samples=self.data[clusters[index]], centroid=prototype)
                for index, prototype in enumerate(prototypes)
            ]
            try:
                selected_distance = np.argmax(
                    [distance_info[0] for index, distance_info in enumerate(not_nearest_centroids)]
                )
                prototypes.append(self.data[not_nearest_centroids[selected_distance][1]])
            except TypeError:
                prototypes = [
                    self._get_random_initial_prototype(), self._get_sample_max_prob_margin()
                ]
        return np.array(prototypes)

    def fit(self):
        """

        :return: fits the training data to the model
        """
        return self.model.fit()

    def get_objective_function(self):
        """

        :return: objective function of the training
        """

        return self.model.get_objective_function()

    def predict(self):
        """

        :return: cluster labels of the given input data
        """
        return self.model.predict()

    def predict_new(self, x):
        """

        :param x: input vector
        :return: cluster label of input vector
        """
        return self.model.predict_new(x)

    def get_centroids(self):
        """

        :return: learned centroids
        """

        return self.model.get_clusters_index_cent()[1]

    def get_distance_space(self, x):
        """

        :param x: input vector
        :return: distance space for the given input vector
        """
        return self.model.get_distance_space(x)

    # classification aspect
    def get_prototypes_class(self, labels):
        """

        :param labels: array-like: labels of the input data set
        :return: component initiatlizer for GNPC classifier designs
        """
        return self.model.get_prototypes(labels)

    def predict_proba(self, x):
        """

        :param x: input vector
        :return: confidence of the given prediction
        """
        return self.model.predict_proba_(x)

    def get_proba(self, x):
        """

        :param x: input vector
        :return: soft confidence for the given prediction
        """
        distance_space = self.get_distance_space(x)
        return np.array([softmin_funct(v) for _, v in enumerate(distance_space)])
