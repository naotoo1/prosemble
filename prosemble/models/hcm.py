"""
Implementation of Hard c Means alternating optimisation algorithm
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT


from collections import Counter

import matplotlib
import numpy as np
# matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
from prosemble.core.distance import (
    euclidean_distance,
    squared_euclidean_distance
)


class Kmeans:
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

    set_prototypes: array-like:
        initial prototypes to  begin with. default is None

    plot_steps: bool:
        True for visulisation of training and False otherwise

    """

    def __init__(self, data, c, num_inter, epsilon, ord, set_prototypes=None, plot_steps=False):
        self.data = data
        self.num_clusters = c
        self.num_iter = num_inter
        self.epsilon = epsilon
        self.set_prototypes = set_prototypes
        self.plot_steps = plot_steps
        self.ord = ord
        self.objective_function = []
        self.fit_cent = []
        self.fit_clus = []

        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)

        if set_prototypes is not None:
            if not isinstance(self.set_prototypes, np.ndarray):
                self.set_prototypes = np.array(self.set_prototypes)
            if self.set_prototypes.shape[1] != self.data.shape[1]:
                raise ValueError(f'The input dim of prototypes {self.set_prototypes.shape[1]} '
                                 f'!= input dim of data {self.data.shape[1]}')
            if len(self.set_prototypes) != self.num_clusters:
                raise ValueError('There should one prototype per class')

    def _select_centroids_randomly(self):
        random_samples_feature_space = np.random.choice(
            self.data.shape[0],
            self.num_clusters, replace=False
        )
        return [self.data[index] for index in random_samples_feature_space]

    def _select_centroids(self):
        if self.set_prototypes is None:
            return self._select_centroids_randomly()
        return self.set_prototypes

    def _initialise_cluster(self, centroids):
        clusters = [[] for _ in range(self.num_clusters)]
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
            [squared_euclidean_distance(sample, centroid) for index, sample in enumerate(samples)]
        )

    @staticmethod
    def _nearest_centroids(sample, centroids):
        return np.argmin([euclidean_distance(sample, centroid) for centroid in centroids])

    def _updated_centroids(self, clusters):
        centroids = np.zeros((self.num_clusters, self.data.shape[1]))
        for cluster_index, cluster in enumerate(clusters):
            try:
                mean_cluster = np.nanmean(self.data[cluster], axis=0)
                centroids[cluster_index] = mean_cluster
            except RuntimeWarning:
                continue
        return centroids

    def _centroid_stability(self, centroids_num, centroids):
        if self.ord is not None and self.epsilon is None:
            distance = np.linalg.norm((centroids_num - centroids), ord=self.ord)
            return np.sum(distance) == 0
        if self.ord is not None and self.epsilon is not None:
            distance = np.linalg.norm((centroids_num - centroids), ord=self.ord)
            return np.sum(distance) <= self.epsilon
        if self.ord is None and self.epsilon is None:
            distance = [euclidean_distance(centroids_num[index], centroids[index]) for index in
                        range(self.num_clusters)]
            return np.sum(distance) == 0
        if self.ord is None and self.epsilon is not None:
            distance = [euclidean_distance(centroids_num[index], centroids[index]) for index in
                        range(self.num_clusters)]
            return np.sum(distance) <= self.epsilon
        return None

    def _get_cluster_results(self, cluster_result):
        labels = np.empty(self.data.shape[0])
        for cluster_index, cluster in enumerate(cluster_result):
            for sample_index in cluster:
                labels[sample_index] = cluster_index
        return labels

    def get_centroids(self):
        centroids = self._select_centroids()
        for num in range(self.num_iter):
            clusters = self._initialise_cluster(centroids)
            self.objective_function.append(self.compute_objective_function(clusters, centroids))
            self.get_plot(clusters, centroids)
            centroids_num = centroids
            centroids = self._updated_centroids(clusters)
            self.get_plot(clusters, centroids)
            if self._centroid_stability(centroids_num, centroids) or num == self.num_iter - 1:
                self.fit_clus.append(clusters)
                self.fit_cent.append(centroids)
                break

    def get_centroids_(self):
        centroids = self._select_centroids()
        optimize = True
        while optimize:
            clusters = self._initialise_cluster(centroids)
            self.objective_function.append(self.compute_objective_function(clusters, centroids))
            self.get_plot(clusters, centroids)
            centroids_num = centroids
            centroids = self._updated_centroids(clusters)
            self.get_plot(clusters, centroids)
            if self._centroid_stability(centroids_num, centroids):
                self.fit_clus.append(clusters)
                self.fit_cent.append(centroids)
                optimize = False

    def get_plot(self, cluster, centroids):
        if self.plot_steps:
            for _, v in enumerate(cluster):
                plt.scatter(self.data[v][:, 0], self.data[v][:, 1])
            for cent in centroids:
                plt.scatter(cent[0], cent[1], marker='v', color='black')
            plt.pause(0.3)
            plt.clf()

    def get_objective_function(self):
        """

        :return: The objective function
        """

        return self.objective_function

    def predict(self):
        """

        :return: The cluster labels for the input data set
        """
        return self._get_cluster_results(self.fit_clus[0])

    def predict_new(self, x):
        """

        :param x: input data
        :return: predicts the cluster label of the input data
        """
        return [
            self._nearest_centroids(sample, self.fit_cent[0]) for index, sample in enumerate(x)
        ]

    def get_clusters_index_cent(self):
        return self.fit_clus, self.fit_cent

    def fit(self):
        """

        :return: fits the model to input data for training
        """
        if self.num_iter is None:
            return self.get_centroids_()
        return self.get_centroids()

    def get_distance_space(self, x):
        """

        :param x: input vector
        :return: distance space for the given input vector
        """
        distance_space = np.array(
            [[euclidean_distance(sample, centroid)
              for centroid in self.get_clusters_index_cent()[1][0]]
             for index, sample in enumerate(x)]
        )
        return distance_space

    def predict_proba_(self, x):
        """

        :param x: input vector
        :return: returns the confidence of the prediction
        """
        init_proba = np.zeros((len(x), self.num_clusters))
        prediction = self.predict_new(x)
        for index, results in enumerate(prediction):
            init_proba[index][results] = 1
        return init_proba

    def get_prototypes(self, labels):
        """

        :param labels: array-like
        :return: prototypes used as component initiatializer  for GNPC classifier design
        """
        self.fit()
        clusters_indices, centroids = self.fit_clus[0], self.fit_cent[0]
        clusters = [labels[cluster_with_indices] for cluster_with_indices in clusters_indices]
        max_occurrence = [dict(Counter(cluster)) for _, cluster in enumerate(clusters)]
        reposition_centroids = np.argsort([max(count, key=count.get) for count in max_occurrence])
        prototypes = centroids[reposition_centroids]
        return prototypes, centroids

    def final_centroids(self):
        """

        :return: learned centroids
        """
        return self.fit_cent[0]
