"""
Implementation of Improved Possibilistic c-Means 2 Alternating Optimization Algorithm
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT


from collections import Counter

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
# matplotlib.use('QtAgg')
from prosemble.core.distance import (
    euclidean_distance,
    squared_euclidean_distance
)
from .fcm import FCM


class IPCM2:
    """
    params:

    data : array-like:
        input data

    c: int:
        number of clusters
    m_f: int:
        fuzzy parameter
    m_p: int:
        possibilistic parameter

    num_iter: int:
        number of iterations

    epsilon: float:
        small difference for termination of algorithm

    ord:  {non-zero int, inf, -inf, ‘fro’, ‘nuc’}
          order of the norm

    set_U_matrix: array-like:
        initial U matrix to  begin with. default is None

    plot_steps: bool:
        True for visualization of training steps and False otherwise
    """

    def __init__(self,
                 data,
                 c,
                 m_f,
                 m_p,
                 num_iter,
                 epsilon,
                 ord,
                 set_centroids=None,
                 set_U_matrix=None,
                 plot_steps=False):
        self.data = data
        self.num_clusters = c
        self.fuzzifier = m_f
        self.tipifier = m_p
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.set_U_matrix = set_U_matrix
        self.set_centroids = set_centroids
        self.plot_steps = plot_steps
        self.ord = ord
        self.objective_function = []
        self.fit_cent = []
        self.fit_clus = []
        self.eta = []

        if self.set_U_matrix == 'fcm':
            self.model1 = FCM(
                data=self.data,
                c=self.num_clusters,
                m=2,
                num_iter=self.num_iter,
                epsilon=self.epsilon,
                ord=self.ord
            )
            self.model1.fit()
            self.set_U_matrix = self.model1.predict_proba_(self.data)
            self.set_centroids = self.model1.final_centroids()

        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)

        if self.set_U_matrix is not None:
            if not isinstance(self.set_U_matrix, np.ndarray):
                self.set_U_matrix = np.array(self.set_U_matrix)
            if self.set_U_matrix.shape[1] != self.num_clusters:
                raise ValueError(f'The input dim of fuzzy U matrix {self.set_U_matrix.shape[1]} '
                                 f'!= number of cluster {self.num_clusters}')
            if len(self.set_U_matrix) != self.data.shape[0]:
                raise ValueError('There should one prototype per class')

    def randomly_initialised_fuzzy_matrix(self):
        return np.random.dirichlet(np.ones(self.num_clusters), size=self.data.shape[0])

    def _select_centroids_randomly(self):
        random_samples_feature_space = np.random.choice(
            self.data.shape[0],
            self.num_clusters,
            replace=False
        )
        return [self.data[index] for index in random_samples_feature_space]

    def compute_centroids(self, fuzzy_matrix, t_matrix):

        fuzzified_assignments = [
            np.power([u_ik[i] for _, u_ik in enumerate(fuzzy_matrix)], self.fuzzifier)
            for i in range(self.num_clusters)
        ]

        tipical_assignments = [
            [t_ik[i] for _, t_ik in enumerate(t_matrix)]
            for i in range(self.num_clusters)
        ]

        prod_assignments = np.multiply(fuzzified_assignments, tipical_assignments)

        sum_assigments = [np.sum(i) for i in prod_assignments]

        centroid_numerator = \
            [[np.multiply(prod_assignments[cluster_index][index], sample) for
              index, sample in enumerate(self.data)] for cluster_index in range(self.num_clusters)]
        centroids = np.array(
            [np.sum(v, axis=0) / sum_assigments[i]
             for i, v in enumerate(centroid_numerator)]
        )
        return centroids

    def compute_gamma_0(self, fuzzy_matrix, centriods):

        fuzzified_assignments = [
            np.power([u_ik[i] for _, u_ik in enumerate(fuzzy_matrix)], self.fuzzifier)
            for i in range(self.num_clusters)
        ]

        sum_fuzzified_assigments = [np.sum(i) for i in fuzzified_assignments]

        centroid_numerator = [
            [np.multiply(fuzzified_assignments[cluster_index][index],
                         squared_euclidean_distance(sample, centriods[cluster_index])) for
             index, sample in enumerate(self.data)] for cluster_index in range(self.num_clusters)]

        gamma = np.array(
            [np.sum(v, axis=0) / sum_fuzzified_assigments[i]
             for i, v in enumerate(centroid_numerator)]
        )

        return gamma

    def compute_gamma_1(self, fuzzy_matrix, t_matrix, centriods):

        fuzzified_assignments = [
            np.power([u_ik[i] for _, u_ik in enumerate(fuzzy_matrix)], self.fuzzifier)
            for i in range(self.num_clusters)
        ]

        tipical_assignments = [
            np.power([t_ik[i] for _, t_ik in enumerate(t_matrix)], self.tipifier)
            for i in range(self.num_clusters)
        ]

        prod_assignment = np.multiply(fuzzified_assignments, tipical_assignments)

        sum_fuzzified_assigments = [np.sum(i) for i in prod_assignment]

        centroid_numerator = [
            [np.multiply(prod_assignment[cluster_index][index],
                         squared_euclidean_distance(sample, centriods[cluster_index])) for
             index, sample in enumerate(self.data)] for cluster_index in range(self.num_clusters)]

        gamma = np.array(
            [np.sum(v, axis=0) / sum_fuzzified_assigments[i]
             for i, v in enumerate(centroid_numerator)]
        )

        return gamma

    def update_tipicality_matrix(self, centroids, g_matrix, t_matrix):
        initial_t_matrix = t_matrix
        for i in range(len(self.data)):
            for j in range(self.num_clusters):
                exponent = np.exp(
                    -(squared_euclidean_distance(centroids[j], self.data[i]) / g_matrix[j]))
                tik_new = exponent
                initial_t_matrix[i][j] = tik_new
        return initial_t_matrix

    def update_fuzzy_matrix(self, centroids, u_matrix, g_matrix):  # equation 14
        initial_u_matrix = u_matrix
        for i in range(len(self.data)):
            denomenator = 0
            for j in range(self.num_clusters):
                den = \
                    (g_matrix[j] *
                     (1 - (np.exp(
                         -(squared_euclidean_distance(centroids[j], self.data[i]) /
                             g_matrix[j])))))
                denomenator += np.power((1 / den), 2 / (self.fuzzifier - 1))
            for j in range(self.num_clusters):
                uik_new = np.power(
                    (1 / (g_matrix[j] *
                          (1 - (np.exp(
                              -(squared_euclidean_distance(centroids[j], self.data[i]) /
                                  g_matrix[j])))))),
                    2 / (self.fuzzifier - 1)) / denomenator
                initial_u_matrix[i][j] = uik_new
        return initial_u_matrix

    def _select_fuzzy_U_matrix(self):
        if self.set_U_matrix is None:
            return self.randomly_initialised_fuzzy_matrix()
        return self.set_U_matrix

    def compute_objective_function_0(self, centroids, t_matrix, fuzzy_matrix):

        objective_objective = \
            np.sum([[squared_euclidean_distance(self.data[i], centroids[j]) *
                     t_matrix[i][j] *
                     np.power(fuzzy_matrix[i][j], self.fuzzifier)
                     for i in range(len(self.data))] for j in range(self.num_clusters)])

        return objective_objective

    def compute_objective_function_1(self, t_matrix, fuzzy_matrix, gamma):
        objective_function = np.sum(
            [gamma[j] * (np.sum(
                [(t_matrix[i][j] * np.log(t_matrix[i][j]) - t_matrix[i][j] + 1) *
                 np.power(fuzzy_matrix[i][j], self.fuzzifier)
                 for i in range(len(self.data))])) for j in range(self.num_clusters)]
        )
        return objective_function

    def compute_objective_function(self, centroids, gamma, t_matrix, fuzzy_matrix):
        return self.compute_objective_function_0(centroids, t_matrix, fuzzy_matrix) + \
            self.compute_objective_function_1(t_matrix, fuzzy_matrix, gamma)

    @staticmethod
    def _distance_(samples, centroid):
        return np.sum(
            [euclidean_distance(sample, centroid) for index, sample in enumerate(samples)]
        )

    @staticmethod
    def _nearest_centroids(sample, centroids):
        return np.argmin([euclidean_distance(sample, centroid) for centroid in centroids])

    def _centroid_stability(self, centroids_num, centroids):
        if self.ord is not None and self.epsilon is None:
            distance = np.linalg.norm((centroids_num - centroids), ord=self.ord)
            return distance == 0
        if self.ord is not None and self.epsilon is not None:
            distance = np.linalg.norm((centroids_num - centroids), ord=self.ord)
            return distance <= self.epsilon
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

    def initialise_cluster(self, centroids):
        clusters = [[] for _ in range(len(centroids))]
        for index, sample in enumerate(self.data):
            centroid_index = self._nearest_centroids(sample, centroids)
            clusters[centroid_index].append(index)
        return clusters

    def get_centroids_0(self):
        u_matrix = self._select_fuzzy_U_matrix()
        centroids = self.set_centroids
        t_matrix = np.zeros((len(self.data), self.num_clusters))
        gamma = self.compute_gamma_0(u_matrix, centroids)
        for num in range(self.num_iter):
            centroids_old = centroids
            clusters = self.initialise_cluster(centroids_old)
            self.get_plot(clusters, centroids_old)
            t_matrix = self.update_tipicality_matrix(centroids_old, gamma, t_matrix)
            u_matrix = self.update_fuzzy_matrix(centroids_old, u_matrix, gamma)
            centroids = self.compute_centroids(u_matrix, t_matrix)
            self.get_plot(cluster=clusters, centroids=centroids)
            if self._centroid_stability(centroids_old, centroids) or num == self.num_iter - 1:
                return u_matrix, t_matrix, centroids

    def get_centroids_1(self):
        u_matrix, t_matrix, centroids = self.get_centroids_0()
        gamma = self.compute_gamma_1(u_matrix, t_matrix, centroids)
        for num in range(self.num_iter):
            centroids_old = centroids
            clusters = self.initialise_cluster(centroids_old)
            self.get_plot(clusters, centroids_old)
            t_matrix = self.update_tipicality_matrix(centroids_old, gamma, t_matrix)
            u_matrix = self.update_fuzzy_matrix(centroids, u_matrix, gamma)
            obj = self.compute_objective_function(centroids, gamma, t_matrix, u_matrix)
            self.objective_function.append(obj)
            centroids = self.compute_centroids(u_matrix, t_matrix)
            self.get_plot(clusters, centroids)
            if self._centroid_stability(centroids_old, centroids) or num == self.num_iter - 1:
                self.fit_clus.append(clusters)
                self.fit_cent.append(centroids)
                self.eta.append(gamma)
                break

    def get_centroids_(self):
        u_matrix = self._select_fuzzy_U_matrix()
        centroids = self.set_centroids
        t_matrix = np.zeros((len(self.data), self.num_clusters))
        gamma = self.compute_gamma_0(u_matrix, centroids)
        optimize = True
        while optimize:
            obj = self.compute_objective_function(centroids, gamma, t_matrix, u_matrix)
            self.objective_function.append(obj)
            clusters = self.initialise_cluster(centroids)
            self.get_plot(clusters, centroids)
            t_matrix = self.update_tipicality_matrix(centroids, gamma, t_matrix)
            u_matrix = self.update_fuzzy_matrix(centroids, u_matrix, gamma)
            centroid_num = centroids
            centroids = self.compute_centroids(u_matrix, t_matrix)
            self.get_plot(clusters, centroids)
            if self._centroid_stability(centroid_num, centroids):
                optimize = False
        return u_matrix, t_matrix, centroids

    def get_centroids_11(self):
        u_matrix, t_matrix, centroids = self.get_centroids_()
        gamma = self.compute_gamma_1(u_matrix, t_matrix, centroids)
        optimize = True
        while optimize:
            clusters = self.initialise_cluster(centroids)
            self.get_plot(clusters, centroids)
            t_matrix = self.update_tipicality_matrix(centroids, gamma, t_matrix)
            u_matrix = self.update_fuzzy_matrix(centroids, u_matrix, gamma)
            centroid_num = centroids
            centroids = self.compute_centroids(u_matrix, t_matrix)
            obj = self.compute_objective_function(centroids, gamma, t_matrix, u_matrix)
            self.objective_function.append(obj)
            self.get_plot(clusters, centroids)
            if self._centroid_stability(centroid_num, centroids):
                self.fit_clus.append(clusters)
                self.fit_cent.append(centroids)
                self.eta.append(gamma)
                optimize = False
        return u_matrix, t_matrix, centroids

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

        :return: array-like: cluster labels of the input dataset
        """
        return self._get_cluster_results(self.fit_clus[0])

    def predict_new(self, x):
        """

        :param x: array-like: input vector
        :return: cluster label of input vector
        """
        return [
            self._nearest_centroids(sample, self.fit_cent[0]) for index, sample in enumerate(x)
        ]

    def get_clusters_index_cent(self):
        return self.fit_clus, self.fit_cent

    def fit(self):
        """

        :return: fits the data set to the model
        """
        if self.num_iter is None:
            return self.get_centroids_11()
        return self.get_centroids_1()

    def get_distance_space(self, x):
        """

        :param x: array-like: input vector
        :return: distance space of the input vector
        """
        distance_space = np.array(
            [[euclidean_distance(sample, centroid)
              for centroid in self.get_clusters_index_cent()[1][0]]
             for index, sample in enumerate(x)]
        )

        return distance_space

    def predict_proba_(self, x):
        """

        :param x:  array-like: input vector
        :return: confidence of predicted label
        """
        initial_u_matrix = np.zeros((len(x), self.num_clusters))
        for i in range(len(x)):
            denomenator = 0
            for j in range(self.num_clusters):
                denomenator += np.power(
                    (1 / (self.eta[0][j] *
                          (1 - np.exp(-(squared_euclidean_distance(
                              self.fit_cent[0][j], self.data[i]) / self.eta[0][j]
                          ))))),
                    2 / (self.fuzzifier - 1))
            for j in range(self.num_clusters):
                uik_new = np.power(
                    (1 / (self.eta[0][j] * (1 - np.exp(-(
                            squared_euclidean_distance(
                                self.fit_cent[0][j], self.data[i]) / self.eta[0][j]))))),
                    2 / (self.fuzzifier - 1)) / denomenator
                initial_u_matrix[i][j] = uik_new
        return initial_u_matrix

    def predict_tipicality_(self, x):
        """

        :param x: array-like: input vector
        :return: tipicality of the input vector
        """
        initial_t_matrix = np.zeros((len(x), self.num_clusters))
        for i in range(len(x)):
            for j in range(self.num_clusters):
                exponent = squared_euclidean_distance(
                    self.fit_cent[0][j], self.data[i]) / self.eta[0][j]
                tik_new = np.exp(-exponent)
                initial_t_matrix[i][j] = tik_new
        return initial_t_matrix

    # Classification aspect
    def get_prototypes(self, labels):
        """

        :param labels: array-like: labels of the imput data set
        :return: prototypes for the GNPC classifier design
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


