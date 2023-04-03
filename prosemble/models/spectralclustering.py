"""
Implementation of spectral clustering with prototype options
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT

from dataclasses import dataclass
# import pandas as pd
import numpy as np

from prosemble.core.graphdata import (
    get_graph_nodes_list,
    preprocess_input_data,
    match_samples_to_nearest_node,
    get_adjacency_matrix,
    get_normalised_laplacian,
    get_affinity_matrix,
    GraphNodeGenerator,
    MatchedSamples,
    PrepareData
)

from .hcm import Kmeans
from .kmeans import kmeans_plusplus
from .fcm import FCM
from .pcm import PCM
from .fpcm import FPCM
from .pfcm import PFCM
from .ipcm import IPCM1
from .ipcm_2 import IPCM2
from .bgpc import BGPC


@dataclass()
class PreprocessInput:
    graph_data: GraphNodeGenerator
    preprocess_data: PrepareData
    matches: MatchedSamples


class SpectralClustering:
    """
    params:
    data : array-like:
        input data

    num_cluster: int:
        number of clusters

    input_dim: int:
        number of features. If None default uses all features of dataset.

    adjacency: array-like:
        adjacency_matrix . If None the graph of the input data is
        learned using Growing Neural Gas Algorithm and onward adjacency computed.

    method: str: {‘hcm’, ‘kmeans_plusplus’, ‘fcm’, ‘pcm’, ‘fpcm’, ‘pfcm’, ‘ipcm’, ‘ipcm2’, ‘bgpc’}
        prototype clustering  default is 'hcm'

    num_iter: int:
        number of iteration

    ord: {non-zero int, inf, -inf, ‘fro’, ‘nuc’}
        order of the norm

    epsilon: float:
        small difference for termination of algorithm

    plot_steps: bool:
        True for visualization of training steps and False otherwise

    """

    def __init__(self, data, input_dim, num_clusters, adjacency, method='hcm',
                 num_iter=1000, epsilon=0.00001, ord=None, plot_steps=False):
        self.data = data
        self.input_dimension: int = input_dim
        self.num_clusters: int = num_clusters
        self.adjacency = adjacency
        self.method: str = method
        self.num_iter = num_iter
        self.epsilon: float = epsilon
        self.ord: str = ord
        self.plot_steps: bool = plot_steps

        if self.input_dimension is None:
            self.input_dimension = np.array(self.data).shape[1]

    def get_inputs(self) -> PreprocessInput:
        graph_data = get_graph_nodes_list(
            self.data,
            self.input_dimension
        )
        preprocess_data = preprocess_input_data(
            graph_data
        )
        matches = match_samples_to_nearest_node(
            preprocess_data,
            graph_data
        )
        return PreprocessInput(
            graph_data=graph_data,
            preprocess_data=preprocess_data,
            matches=matches
        )

    def affinity(self, inputs: PreprocessInput):
        if self.adjacency is None:
            adjacency_matrix = get_adjacency_matrix(
                inputs.matches,
                inputs.graph_data
            )
            affinity_matrix = get_affinity_matrix(
                adjacency_matrix
            )
        else:
            adjacency_matrix = self.adjacency
            affinity_matrix = get_affinity_matrix(
                adjacency_matrix
            )
        return affinity_matrix

    def get_laplacian_(self) -> np.ndarray:
        affinity_matrix = self.affinity(
            self.get_inputs()
        )
        return get_normalised_laplacian(
            affinity_matrix
        )

    def get_n_eigen_values(self):
        number_of_nodes = self.get_laplacian_().shape[0]
        cluster_number = self.num_clusters
        u, s, vN = np.linalg.svd(
            self.get_laplacian_(),
            full_matrices=False
        )
        vN = np.transpose(vN)
        eigen_vectors = \
            vN[:, number_of_nodes - cluster_number:number_of_nodes]
        return eigen_vectors

    def fit_method(self):
        data = np.array(self.get_n_eigen_values())

        if self.method == 'hcm':
            kmeans = Kmeans(
                data=data,
                c=self.num_clusters,
                epsilon=self.epsilon,
                ord=self.ord,
                num_inter=self.num_iter,
                plot_steps=self.plot_steps
            )

            kmeans.fit()

            labels = kmeans.predict()
            return labels

        if self.method == 'kmeans_plusplus':
            kmeans_plus = kmeans_plusplus(
                data=data,
                c=self.num_clusters,
                epsilon=self.epsilon,
                ord=self.ord,
                num_inter=self.num_iter,
                plot_steps=self.plot_steps
            )

            kmeans_plus.fit()
            labels_plus = kmeans_plus.predict()
            return labels_plus

        if self.method == 'fcm':
            fcm = FCM(
                data=data,
                c=self.num_clusters,
                epsilon=self.epsilon,
                ord=self.ord,
                num_iter=self.num_iter,
                m=2,
                plot_steps=self.plot_steps,
            )

            fcm.fit()
            labels_plus = fcm.predict()
            return labels_plus

        if self.method == 'pcm':
            pcm = PCM(
                data=data,
                c=self.num_clusters,
                epsilon=self.epsilon,
                ord=self.ord,
                num_iter=self.num_iter,
                k=0.01,
                m=2,
                set_U_matrix='fcm',
                plot_steps=self.plot_steps,
            )

            pcm.fit()
            labels_plus = pcm.predict()
            return labels_plus

        if self.method == 'fpcm':
            fpcm = FPCM(
                data=data,
                c=self.num_clusters,
                epsilon=self.epsilon,
                ord=self.ord,
                num_iter=self.num_iter,
                m=2,
                eta=2,
                plot_steps=self.plot_steps
            )

            fpcm.fit()
            labels_plus = fpcm.predict()

            return labels_plus

        if self.method == 'pfcm':
            pfcm = PFCM(
                data=data,
                c=self.num_clusters,
                num_iter=self.num_iter,
                epsilon=self.epsilon,
                k=2,
                ord=self.ord,
                set_U_matrix='fcm',
                a=2,
                b=2,
                eta=2,
                m=2,
                plot_steps=self.plot_steps
            )

            pfcm.fit()
            labels_pfcm = pfcm.predict()
            return labels_pfcm

        if self.method == 'ipcm':
            ipcm = IPCM1(
                data=data,
                c=self.num_clusters,
                num_iter=None,
                epsilon=self.epsilon,
                m_f=2,
                m_p=2,
                k=2,
                ord=self.ord,
                set_U_matrix='fcm',
                plot_steps=self.plot_steps
            )

            ipcm.fit()
            labels_ipcm = ipcm.predict()

            return labels_ipcm

        if self.method == 'ipcm2':
            ipcm2 = IPCM2(
                data=data,
                c=self.num_clusters,
                num_iter=self.num_iter,
                epsilon=self.epsilon,
                m_f=2,
                m_p=2,
                ord=self.ord,
                set_U_matrix='fcm',
                plot_steps=self.plot_steps
            )

            ipcm2.fit()
            labels_ipcm2 = ipcm2.predict()
            return labels_ipcm2

        if self.method == 'bgpc':
            bgpc = BGPC(
                data=data,
                c=self.num_clusters,
                num_iter=self.num_iter,
                epsilon=self.epsilon,
                a_f=2,
                b_f=2,
                ord=self.ord,
                set_U_matrix='fcm',
                plot_steps=self.plot_steps
            )

            bgpc.fit()
            labels_ipcm2 = bgpc.predict()
            return labels_ipcm2
        raise RuntimeError(
            "fit_method:none of the checks did match"
        )

    def predict(self):
        return self.fit_method()
