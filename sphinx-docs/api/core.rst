Core API Reference
==================

Core modules provide the building blocks used by all models:
distance functions, loss functions, initializers, and utilities.

Distance Functions
------------------

.. automodule:: prosemble.core.distance
   :members:
   :undoc-members:

Similarities
------------

.. automodule:: prosemble.core.similarities
   :members:
   :undoc-members:

Kernel Functions
-----------------

.. automodule:: prosemble.core.kernel
   :members:
   :undoc-members:

Loss Functions
--------------

.. automodule:: prosemble.core.losses
   :members:
   :undoc-members:

Activations
-----------

.. automodule:: prosemble.core.activations
   :members:
   :undoc-members:

Competitions
------------

.. automodule:: prosemble.core.competitions
   :members:
   :undoc-members:

Initializers
------------

.. automodule:: prosemble.core.initializers
   :members:
   :undoc-members:

Pooling
-------

.. automodule:: prosemble.core.pooling
   :members:
   :undoc-members:

Quantization
------------

.. automodule:: prosemble.core.quantization
   :members:
   :undoc-members:

Utilities
---------

.. automodule:: prosemble.core.utils
   :members:
   :undoc-members:

Pipeline
--------

.. autoclass:: prosemble.core.pipeline.NotFittedError
   :members:
   :undoc-members:

.. autoclass:: prosemble.core.pipeline.TransformerMixin
   :members:
   :undoc-members:

.. autoclass:: prosemble.core.pipeline.StandardScaler
   :members: fit, transform, fit_transform, get_params, set_params
   :undoc-members:

.. autoclass:: prosemble.core.pipeline.MinMaxScaler
   :members: fit, transform, fit_transform, get_params, set_params
   :undoc-members:

.. autoclass:: prosemble.core.pipeline.PCA
   :members: fit, transform, fit_transform, get_params, set_params
   :undoc-members:

.. autoclass:: prosemble.core.pipeline.Pipeline
   :members: fit, predict, predict_proba, transform, fit_transform, get_params, set_params
   :undoc-members:

Model Selection
---------------

.. autofunction:: prosemble.core.model_selection.clone

.. autofunction:: prosemble.core.model_selection.cross_val_score

.. autoclass:: prosemble.core.model_selection.GridSearchCV
   :members: fit, predict, predict_proba
   :undoc-members:

Evaluation
----------

.. automodule:: prosemble.core.evaluation
   :members: rank_methods, friedman_test, nemenyi_critical_difference, holm_correction, compare_to_reference, expected_calibration_error, selective_accuracy

.. autoclass:: prosemble.core.evaluation.FriedmanResult

.. autoclass:: prosemble.core.evaluation.PairwiseResult

Datasets
--------

.. automodule:: prosemble.datasets.dataset
   :members:
   :undoc-members:
   :no-index:

ONNX Export
-----------

.. automodule:: prosemble.core.onnx_export
   :members: export_onnx
   :undoc-members:

Soft Dynamic Time Warping
-------------------------

.. automodule:: prosemble.core.soft_dtw
   :members: soft_dtw, dtw, soft_dtw_divergence, soft_dtw_matrix, dtw_matrix, soft_dtw_divergence_matrix

Manifolds
---------

.. automodule:: prosemble.core.manifolds
   :members:
   :undoc-members:

Riemannian Optimisation
-----------------------

.. automodule:: prosemble.core.riemannian_optim
   :members: riemannian_update, supports_riemannian_update

Protocols
---------

.. automodule:: prosemble.core.protocols
   :members:
   :undoc-members:

Distributed Training
--------------------

.. automodule:: prosemble.core.distributed
   :members:
   :undoc-members:

Data Utilities
--------------

.. automodule:: prosemble.core.data
   :members:
   :undoc-members:
   :no-index:

Custom Optimizers
-----------------

.. automodule:: prosemble.core.optimizers
   :members:
   :undoc-members:

Regularization
--------------

.. automodule:: prosemble.core.regularization
   :members:
   :undoc-members:

Curriculum Learning
-------------------

.. automodule:: prosemble.core.curriculum
   :members:
   :undoc-members:

Reject Option
-------------

.. automodule:: prosemble.core.reject
   :members:
   :undoc-members:

Geodesic Interpolation
----------------------

.. automodule:: prosemble.core.geodesic
   :members:
   :undoc-members:
