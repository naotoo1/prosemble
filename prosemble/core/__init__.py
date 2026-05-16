"""
Prosemble core - JAX-based implementations
"""

# Import from JAX modules
from .distance import *
from .kernel import *
from .utils import *
from .initializers import *
from .visualizer import LiveVisualizer
from .callbacks import Callback, VisualizationCallback
from .quantization import MetadataCollectorMixin, QuantizationMixin
from .vis import (
    plot_umatrix, plot_hit_map, plot_component_planes, plot_som_grid,
    plot_som_loss, plot_som_summary,
    plot_decision_boundary_2d, plot_prototype_trajectory, plot_relevance_matrix,
    plot_lvq_summary,
    plot_neural_gas,
)

# Phase 1 core primitives
from .activations import identity, sigmoid_beta, swish_beta, get_activation
from .pooling import (
    stratified_min_pooling, stratified_sum_pooling,
    stratified_max_pooling, stratified_prod_pooling,
)
from .competitions import wtac, knnc, cbcc
from .similarities import gaussian_similarity, cosine_similarity_matrix, euclidean_similarity, rank_scaled_gaussian
from .losses import (
    glvq_loss, glvq_loss_with_transfer, lvq1_loss, lvq21_loss,
    nllr_loss, rslvq_loss, cross_entropy_lvq_loss, margin_loss,
    neural_gas_energy,
)
from .pipeline import Pipeline, StandardScaler, MinMaxScaler, PCA, TransformerMixin
from .model_selection import GridSearchCV, cross_val_score, clone
from .protocols import (
    Manifold, CallbackLike,
    DistanceMatrixFn, DistancePairwiseFn,
    SupervisedInitFn, UnsupervisedInitFn,
)
from .serialization import SerializationMixin
from .data import shuffle_arrays, padded_batches, batched_iterator
from .distributed import create_mesh, shard_data, replicate_params

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    import warnings
    warnings.warn("JAX is required for prosemble. Install with: pip install jax")

from .distance import __all__ as _distance_all

__all__ = _distance_all + [
    'LiveVisualizer',
    'Callback',
    'VisualizationCallback',
    # Visualization
    'plot_umatrix', 'plot_hit_map', 'plot_component_planes', 'plot_som_grid',
    'plot_som_loss', 'plot_som_summary',
    'plot_decision_boundary_2d', 'plot_prototype_trajectory', 'plot_relevance_matrix',
    'plot_lvq_summary',
    'plot_neural_gas',
    'MetadataCollectorMixin',
    'QuantizationMixin',
    'JAX_AVAILABLE',
    # Activations
    'identity', 'sigmoid_beta', 'swish_beta', 'get_activation',
    # Pooling
    'stratified_min_pooling', 'stratified_sum_pooling',
    'stratified_max_pooling', 'stratified_prod_pooling',
    # Competitions
    'wtac', 'knnc', 'cbcc',
    # Similarities
    'gaussian_similarity', 'cosine_similarity_matrix', 'euclidean_similarity',
    'rank_scaled_gaussian',
    # Losses
    'glvq_loss', 'glvq_loss_with_transfer', 'lvq1_loss', 'lvq21_loss',
    'nllr_loss', 'rslvq_loss', 'cross_entropy_lvq_loss', 'margin_loss',
    'neural_gas_energy',
    # Pipeline & Transformers
    'Pipeline', 'StandardScaler', 'MinMaxScaler', 'PCA', 'TransformerMixin',
    # Model Selection
    'GridSearchCV', 'cross_val_score', 'clone',
    # Protocols & type aliases
    'Manifold', 'CallbackLike',
    'DistanceMatrixFn', 'DistancePairwiseFn',
    'SupervisedInitFn', 'UnsupervisedInitFn',
    # Serialization
    'SerializationMixin',
    # Data loading utilities
    'shuffle_arrays', 'padded_batches', 'batched_iterator',
    # Distributed training
    'create_mesh', 'shard_data', 'replicate_params',
    # ONNX export (optional)
    'export_onnx',
]

# Conditional ONNX export
try:
    from .onnx_export import export_onnx
except ImportError:
    pass
