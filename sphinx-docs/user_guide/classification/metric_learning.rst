Metric learning
===============

.. currentmodule:: prosemble.models

A fixed Euclidean distance treats every feature as equally important and
independent. Metric-learning models learn the distance together with the
prototypes. The learned metric is itself an explanation: it shows which
features, and which combinations of features, separate the classes.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Model
     - Distance
     - Learned in addition to the prototypes
   * - :class:`GRLVQ`
     - :math:`\sum_j \lambda_j (x_j - w_j)^2`
     - One relevance per feature
   * - :class:`GMLVQ`
     - :math:`\|\Omega (x - w)\|^2`
     - One matrix shared by all prototypes
   * - :class:`LGMLVQ`
     - :math:`\|\Omega_k (x - w_k)\|^2`
     - One matrix per prototype
   * - :class:`GTLVQ`
     - :math:`\|(I - \Omega_k \Omega_k^\top)(x - w_k)\|^2`
     - An invariance subspace per prototype

See :doc:`/user_guide/interpretation` for how to read relevances and
matrices, and :doc:`/tutorials/interpretable_classification` for a worked
example.

GRLVQ — Feature Relevances
--------------------------

Adds per-feature relevance weights :math:`\lambda_j` that reveal which
features matter for classification.

.. code-block:: python

   from prosemble.models import GRLVQ

   model = GRLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

   # Inspect learned relevances
   print(model.relevances_)  # shape: (n_features,)

The relevance vector satisfies :math:`\lambda_j \geq 0` and
:math:`\sum_j \lambda_j = 1`. Features with large :math:`\lambda_j` are
important for classification.

GMLVQ — Matrix Metric Learning
------------------------------

Learns a full linear transformation :math:`\Omega` such that the distance
is :math:`d(x, w) = \|\Omega(x - w)\|^2`. This captures feature
correlations.

.. code-block:: python

   from prosemble.models import GMLVQ

   model = GMLVQ(
       n_prototypes_per_class=1,
       latent_dim=2,        # project to 2D
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)

   # Learned matrices
   print(model.omega_matrix.shape)   # (latent_dim, n_features)
   print(model.lambda_matrix)        # Omega^T @ Omega — feature importance

   # Feature importance from diagonal
   import jax.numpy as jnp
   feature_importance = jnp.diag(model.lambda_matrix)

When ``latent_dim < n_features``, :math:`\Omega` also performs dimensionality
reduction optimized for classification.

LGMLVQ — Local Metrics
----------------------

Each prototype gets its own :math:`\Omega_k` matrix, allowing different
regions of the feature space to use different metrics.

.. code-block:: python

   from prosemble.models import LGMLVQ

   model = LGMLVQ(
       n_prototypes_per_class=2,
       latent_dim=2,
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)

GTLVQ — Tangent Distance
------------------------

Learns per-prototype invariance subspaces. The tangent distance measures
distance only in directions orthogonal to the invariance subspace:
:math:`d(x, w_k) = \|(I - \Omega_k \Omega_k^T)(x - w_k)\|^2`.

.. code-block:: python

   from prosemble.models import GTLVQ

   model = GTLVQ(
       n_prototypes_per_class=2,
       subspace_dim=1,       # 1D invariance subspace
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)
