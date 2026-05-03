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

Datasets
--------

.. automodule:: prosemble.datasets.dataset
   :members:
   :undoc-members:
