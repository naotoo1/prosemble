Core
====

Building blocks used by the models. Each module is importable from
:mod:`prosemble.core`, for example ``from prosemble.core.distance import
omega_distance``.

Distances and similarities
--------------------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.distance <core/distance>`
     - JAX-based distance functions for Prosemble.
   * - :doc:`prosemble.core.similarities <core/similarities>`
     - Similarity functions for prototype-based learning.
   * - :doc:`prosemble.core.kernel <core/kernel>`
     - JAX-based kernel functions for kernel clustering methods.
   * - :doc:`prosemble.core.soft_dtw <core/soft_dtw>`
     - Soft dynamic time warping.

Losses, activations and competitions
------------------------------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.losses <core/losses>`
     - Loss functions for prototype-based learning.
   * - :doc:`prosemble.core.activations <core/activations>`
     - Transfer/activation functions for prototype-based learning.
   * - :doc:`prosemble.core.competitions <core/competitions>`
     - Competition mechanisms for prototype-based classification.
   * - :doc:`prosemble.core.pooling <core/pooling>`
     - Stratified pooling operations for prototype-based learning.

Initialization
--------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.initializers <core/initializers>`
     - Prototype and parameter initializers for prototype-based learning.

Manifolds and Riemannian optimization
-------------------------------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.manifolds <core/manifolds>`
     - Riemannian manifold primitives for prototype learning.
   * - :doc:`prosemble.core.riemannian_optim <core/riemannian_optim>`
     - Riemannian optimisation for manifold-valued prototypes.
   * - :doc:`prosemble.core.geodesic <core/geodesic>`
     - Geodesic interpolation and boundary visualization for Riemannian models.

Optimization and regularization
-------------------------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.optimizers <core/optimizers>`
     - Custom optax-compatible optimizers for prototype-based learning.
   * - :doc:`prosemble.core.regularization <core/regularization>`
     - Regularization techniques for prototype-based models.
   * - :doc:`prosemble.core.curriculum <core/curriculum>`
     - Curriculum learning (self-paced learning) for prototype-based models.

Reliability and evaluation
--------------------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.reject <core/reject>`
     - Reject option with calibrated uncertainty for prototype classifiers.
   * - :doc:`prosemble.core.evaluation <core/evaluation>`
     - Statistics for comparing classifiers over several datasets.

Pipelines and model selection
-----------------------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.pipeline <core/pipeline>`
     - JAX-native Pipeline and Transformers for prosemble.
   * - :doc:`prosemble.core.model_selection <core/model_selection>`
     - JAX-native model selection: clone, cross_val_score, GridSearchCV.

Data handling and scaling
-------------------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.data <core/data>`
     - Data loading and batching utilities for prosemble.
   * - :doc:`prosemble.core.distributed <core/distributed>`
     - Distributed training utilities for multi-device data parallelism.

Persistence and export
----------------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.serialization <core/serialization>`
     - Unified save/load serialization for all prosemble model families.
   * - :doc:`prosemble.core.quantization <core/quantization>`
     - Shared mixins for model base classes.
   * - :doc:`prosemble.core.onnx_export <core/onnx_export>`
     - ONNX export for prosemble prototype-based models.

Callbacks and visualization
---------------------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.callbacks <core/callbacks>`
     - Callback system for clustering model training.
   * - :doc:`prosemble.core.vis <core/vis>`
     - Visualization utilities for prosemble models.
   * - :doc:`prosemble.core.visualizer <core/visualizer>`
     - Live visualization utility for JAX clustering models.

Utilities and protocols
-----------------------

.. list-table::
   :widths: 30 70

   * - :doc:`prosemble.core.utils <core/utils>`
     - JAX utility functions for data preprocessing and manipulation.
   * - :doc:`prosemble.core.protocols <core/protocols>`
     - Structural typing protocols for prosemble interfaces.

.. toctree::
   :hidden:

   core/distance
   core/similarities
   core/kernel
   core/soft_dtw
   core/losses
   core/activations
   core/competitions
   core/pooling
   core/initializers
   core/manifolds
   core/riemannian_optim
   core/geodesic
   core/optimizers
   core/regularization
   core/curriculum
   core/reject
   core/evaluation
   core/pipeline
   core/model_selection
   core/data
   core/distributed
   core/serialization
   core/quantization
   core/onnx_export
   core/callbacks
   core/vis
   core/visualizer
   core/utils
   core/protocols
