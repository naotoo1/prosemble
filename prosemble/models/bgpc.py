"""
Implementation of Basic Graded Possibilistic alternating optimisation algorithm
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT


from collections import Counter

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

# matplotlib.use('QtAgg')
from .fcm import FCM

from prosemble.core.distance import euclidean_distance

np.seterr(all='ignore')


class BGPC:
    """
    params:

    data : array-like:
        input data

    c: int:
        number of clusters

    num_iter: int:
        number of iterations
    a_f: float:
        alpha parameter

    b_f: float
        beta parameter

    epsilon: float:
        small difference for termination of algorithm

    ord:  {non-zero int, inf, -inf, ‘fro’, ‘nuc’}
          order of the norm

    set_U_matrix: array-like:
        initial U matrix to  begin with. default is None

    plot_steps: bool:
        True for visualization of training steps and False otherwise
    """

    def __init__(self, data, c, num_iter, a_f, b_f, epsilon, ord, set_centroids=None,
                 set_U_matrix=None,
                 plot_steps=False):
        self.data = data
        self.num_clusters = c
        self.a_f = a_f
        self.b_f = b_f
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.set_U_matrix = set_U_matrix
        self.set_centroids = set_centroids
        self.plot_steps = plot_steps
        self.ord = ord
        self.objective_function = []
        self.fit_cent = []
        self.fit_clus = []
        self.b = []
        self.a = []

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

    def compute_centroids(self, u_matrix):

        fuzzified_assignments = [
            [u_ik[i] for _, u_ik in enumerate(u_matrix)]
            for i in range(self.num_clusters)
        ]

        sum_fuzzified_assigments = [np.sum(i) for i in fuzzified_assignments]

        centroid_numerator = [
            [np.multiply(fuzzified_assignments[cluster_index][index], sample)
             for index, sample in enumerate(self.data)]
            for cluster_index in range(self.num_clusters)
        ]

        centroids = np.array(
            [np.sum(v, axis=0) / sum_fuzzified_assigments[i]
             for i, v in enumerate(centroid_numerator)]
        )

        return centroids

    def compute_beta_decay(self, iter):
        return 0.1 * np.power((self.b_f / 0.1), (iter / self.num_iter))

    def compute_alpha_decay(self, iter):
        return (1 - self.b_f) * (1 + np.exp(iter - self.num_iter) + self.a_f)

    def compute_v_matrix(self, centroids, b, v_matrix):
        initial_v_matrix = v_matrix
        for i in range(len(self.data)):
            for j in range(self.num_clusters):
                exponent = np.exp((-euclidean_distance(centroids[j], self.data[i]) / b))
                vik_new = exponent
                initial_v_matrix[i][j] = vik_new
        return initial_v_matrix

    @staticmethod
    def compute_z_list(v_matrix, a):
        z_k = []
        for _, v_ik in enumerate(v_matrix):
            if np.sum(np.power(v_ik, 1 / a)) > 1:
                z_k.append(np.power(np.sum(np.power(v_ik, 1 / a)), a))
            if np.sum(np.power(v_ik, a)) < 1:
                z_k.append(np.power(np.sum(np.power(v_ik, a)), 1 / a))
            if np.sum(np.power(v_ik, a)) == 1:
                z_k.append(1)
            if np.sum(np.power(v_ik, 1 / a)) == 1:
                z_k.append(1)
        return z_k

    def update_u_matrix(self, v_matrix, z_list):
        initial_u_matrix = v_matrix
        for i in range(len(self.data)):
            for j in range(self.num_clusters):
                try:
                    uik_new = v_matrix[i][j] / z_list[i]
                    initial_u_matrix[i][j] = uik_new
                except IndexError:
                    pass
        return initial_u_matrix

    def _select_fuzzy_U_matrix(self):
        if self.set_U_matrix is None:
            return self.randomly_initialised_fuzzy_matrix()
        return self.set_U_matrix

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

    def centroids_init_option(self):
        if self.set_centroids is None:
            return self._select_centroids_randomly()
        return self.set_centroids

    def get_centroids_(self):
        centroids = self.centroids_init_option()
        v_matrix = np.zeros((len(self.data), self.num_clusters))
        for num in range(self.num_iter):
            centroids_old = centroids
            clusters = self.initialise_cluster(centroids_old)
            self.get_plot(clusters, centroids_old)
            b = self.compute_beta_decay(iter=num)
            v_matrix = self.compute_v_matrix(centroids, b, v_matrix)
            a = self.compute_alpha_decay(iter=num)
            z_list = self.compute_z_list(v_matrix, a)
            u_matrix = self.update_u_matrix(v_matrix, z_list)
            centroids = self.compute_centroids(u_matrix)
            self.get_plot(clusters, centroids)
            if self._centroid_stability(centroids_old, centroids) or num == self.num_iter - 1:
                self.fit_clus.append(clusters)
                self.fit_cent.append(centroids)
                self.b.append(b)
                self.a.append(a)
                break

    def get_plot(self, cluster, centroids):
        if self.plot_steps:
            for _, v in enumerate(cluster):
                plt.scatter(self.data[v][:, 0], self.data[v][:, 1])
            for cent in centroids:
                plt.scatter(cent[0], cent[1], marker='v', color='black')
            plt.pause(0.3)
            plt.clf()

    def predict(self):
        """

        :return: array-like: cluster labels of input data
        """
        return self._get_cluster_results(self.fit_clus[0])

    def predict_new(self, x):
        """

        :param x: array-like: input vector
        :return: cluster label of intput vector
        """
        return [
            self._nearest_centroids(sample, self.fit_cent[0])
            for index, sample in enumerate(x)
        ]

    def get_clusters_index_cent(self):
        return self.fit_clus, self.fit_cent

    def fit(self):
        """

        :return: fits the model to the input data set
        """
        return self.get_centroids_()

    def get_distance_space(self, x):

        """

        :param x: array-like: input vector
        :return:  distance space of the input vector
        """

        distance_space = np.array(
            [[euclidean_distance(sample, centroid)
              for centroid in self.get_clusters_index_cent()[1][0]]
             for index, sample in enumerate(x)]
        )

        return distance_space

    def predict_tipicality(self, x):
        """

        :param x: array-like: input vector
        :return: confidence of the input vector
        """
        initial_v_matrix = np.zeros((len(x), self.num_clusters))
        for i in range(len(x)):
            for j in range(self.num_clusters):
                exponent = np.exp(-(euclidean_distance(self.fit_cent[0][j], x[i]) / self.b[0]))
                vik_new = exponent
                initial_v_matrix[i][j] = vik_new
        return initial_v_matrix

    def predict_proba(self, x):
        """

        :param x: array-like: input vector
        :return:  confidence of the input vector
        """
        v_matrix = self.predict_tipicality(x)
        z = self.compute_z_list(v_matrix=v_matrix, a=self.a[0])
        for i in range(len(x)):
            for j in range(self.num_clusters):
                try:
                    uik_new = v_matrix[i][j] / z[i]
                    v_matrix[i][j] = uik_new
                except IndexError:
                    pass
        return v_matrix

    # Classification aspect
    def get_prototypes(self, labels):
        """

        :param labels: labels of the input data set
        :return: array-like: components needed for GNPC classifier design
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
