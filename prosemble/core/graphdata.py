
from dataclasses import dataclass

import numpy as np
from neupy import algorithms
from neupy.algorithms.competitive import growing_neural_gas
from prosemble.models.knn import KNN

import pandas as pd


@dataclass
class PrepareData:
    train_data: pd.DataFrame
    train_labels: pd.DataFrame
    test_data: pd.DataFrame


@dataclass
class MatchedSamples:
    matched_samples: pd.DataFrame


@dataclass
class AffinityMatrix:
    affinity_matrix: pd.DataFrame
    degree_matrix: np.ndarray


@dataclass
class NormalisedLaplacian:
    normalised_laplacian: np.ndarray


@dataclass
class Laplacian:
    laplacian_matrix: np.ndarray


@dataclass
class AdjacencyMatrix:
    adjacency_matrix: pd.DataFrame


@dataclass
class GraphNodeGenerator:
    learned_nodes_list: list
    number_of_features: int
    learned_graph: growing_neural_gas.NeuralGasGraph
    data: pd.DataFrame
    graph_data: pd.DataFrame


def get_graph_nodes_list(
        data,
        num_features: int,
        epochs: int = 100) -> GraphNodeGenerator:
    gng = algorithms.GrowingNeuralGas(
        n_inputs=num_features,
        n_start_nodes=5,
        shuffle_data=True,
        verbose=False,
        step=0.1,
        neighbour_step=0.01,
        max_edge_age=50,
        max_nodes=100,
        n_iter_before_neuron_added=100,
        after_split_error_decay_rate=0.5,
        error_decay_rate=0.995,
        min_distance_for_update=0.05
    )

    gng.train(data, epochs=epochs)
    g = gng.graph
    node_list = g.nodes

    node_with_index = []
    for node in gng.graph.nodes:
        row = node.weight[0].tolist()
        row.append(node_list.index(node))
        node_with_index.append(row)

    train_data = pd.DataFrame(node_with_index)

    return GraphNodeGenerator(
        learned_nodes_list=node_list,
        number_of_features=num_features,
        learned_graph=g,
        data=data,
        graph_data=train_data
    )


def preprocess_input_data(input_data: GraphNodeGenerator) -> PrepareData:
    train_data_ = input_data.graph_data.iloc[:, :input_data.number_of_features]
    train_data_labels = input_data.graph_data.iloc[:, input_data.number_of_features]
    test_data = pd.DataFrame(input_data.data).iloc[:, :input_data.number_of_features]

    return PrepareData(
        train_data=train_data_,
        train_labels=train_data_labels,
        test_data=test_data
    )


def match_samples_to_nearest_node(
        input_data: PrepareData,
        number_of_features: GraphNodeGenerator) -> MatchedSamples:
    scaled_sample_node_matching = []

    knn = KNN(
        dataset=np.array(input_data.train_data),
        labels=input_data.train_labels,
        c=1)

    predict = knn.predict(
        np.array(input_data.test_data)
    )

    for index, data in enumerate(np.array(input_data.test_data)):
        row = data.tolist()
        row.append(predict[index])
        scaled_sample_node_matching.append(row)

    # Convert series into dataframe
    scaled_sample_node_matching = pd.DataFrame(scaled_sample_node_matching)
    scaled_sample_node_matching = scaled_sample_node_matching.rename(
        columns={number_of_features.number_of_features: 'node'}
    )

    return MatchedSamples(
        matched_samples=scaled_sample_node_matching
    )


def get_adjacency_matrix(
        matched_samples: MatchedSamples,
        node_list: GraphNodeGenerator) -> AdjacencyMatrix:
    n = len(node_list.learned_nodes_list)
    matrix_transition = np.zeros((n, n))
    previous_node = -1

    for i, row in matched_samples.matched_samples.iterrows():
        current_node = int(row['node'])
        if previous_node != -1:
            matrix_transition[previous_node, current_node] = \
                matrix_transition[previous_node, current_node] + 1
        else:
            pass
        previous_node = current_node

    adjacency_matrix = pd.DataFrame(matrix_transition)

    return AdjacencyMatrix(
        adjacency_matrix=adjacency_matrix
    )


def get_affinity_matrix(adjacency_matrix: AdjacencyMatrix) -> AffinityMatrix:
    # Sum the rows of the adjacency matrix
    adjacency_mat = adjacency_matrix.adjacency_matrix
    adjacency_mat['sum_of_rows'] = adjacency_mat.sum(axis=1)
    list_degree = adjacency_mat['sum_of_rows'].tolist()

    # Degree matrix of the updated edges
    degree_matrix = np.diag(list_degree)

    # Compute transition probability from node to node
    df_affinity = adjacency_mat.loc[:, :].div(adjacency_mat["sum_of_rows"], axis=0)
    df_affinity = df_affinity.drop(["sum_of_rows"], axis=1)
    df_affinity = df_affinity.fillna(0)

    return AffinityMatrix(
        affinity_matrix=df_affinity,
        degree_matrix=degree_matrix
    )


def get_affinity_matrix1(adjacency_matrix: AdjacencyMatrix) -> AffinityMatrix:
    # Sum the rows of the adjacency matrix
    adjacency_mat = adjacency_matrix.adjacency_matrix
    adjacency_mat['sum_of_rows'] = adjacency_mat.sum(axis=1)
    list_degree = adjacency_mat['sum_of_rows'].tolist()

    # Degree matrix of the updated edges
    degree_matrix = np.diag(list_degree)

    # Compute transition probability from node to node
    df_affinity = adjacency_mat.loc[:, :].div(adjacency_mat["sum_of_rows"], axis=0)
    df_affinity = df_affinity.drop(["sum_of_rows"], axis=1)
    df_affinity = df_affinity.fillna(0)

    return AffinityMatrix(
        affinity_matrix=df_affinity,
        degree_matrix=degree_matrix
    )


def get_normalised_laplacian(affinity_matrix: AffinityMatrix) -> np.ndarray:
    # Compute the normalized laplacian
    I = np.identity(affinity_matrix.affinity_matrix.shape[0])
    sqrt = np.sqrt(affinity_matrix.degree_matrix)
    D_inv_sqrt = np.linalg.pinv(sqrt)
    normalised_laplace = I - np.dot(D_inv_sqrt, affinity_matrix.affinity_matrix).dot(D_inv_sqrt)

    return normalised_laplace


def get_n_eigen_vectors(affinity_matrix: pd.DataFrame,
                        laplacian_matrix: pd.DataFrame,
                        cluster_numer: int):
    # Compute the n smallest eigen vectors of the affinity matrix
    number_of_nodes = affinity_matrix.affinity_matrix.shape[0]
    cluster_number = cluster_numer
    u, s, vN = np.linalg.svd(laplacian_matrix, full_matrices=False)
    vN = np.transpose(vN)
    eigen_vectors = vN[:, number_of_nodes - cluster_number:number_of_nodes]
    return eigen_vectors
