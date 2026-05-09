Supervised Models
=================

All supervised models in prosemble follow the same ``fit`` / ``predict`` API.
They differ in the distance metric, loss function, and representation.

GLVQ — Baseline
----------------

Generalized Learning Vector Quantization is the foundation model. It
learns prototype positions to minimize the relative distance cost
:math:`\mu = (d^+ - d^-) / (d^+ + d^-)`.

.. code-block:: python

   from prosemble.models import GLVQ
   from prosemble.datasets import load_iris_jax
   from prosemble.core.utils import train_test_split_jax

   dataset = load_iris_jax()
   X, y = dataset.input_data, dataset.labels
   X_train, X_test, y_train, y_test = train_test_split_jax(X, y)

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X_train, y_train)

   predictions = model.predict(X_test)
   probabilities = model.predict_proba(X_test)

**Unequal prototypes per class:**

.. code-block:: python

   model = GLVQ(
       n_prototypes_per_class={0: 3, 1: 2, 2: 1},
       prototypes_initializer='class_conditional_mean',
   )

**Prototype initializers** (pass as string):

- ``'stratified_random'`` — random samples per class (default)
- ``'class_mean'`` — class centroids
- ``'class_conditional_mean'`` — class centroids replicated
- ``'stratified_noise'`` — random samples + Gaussian noise
- ``'random_normal'`` — random normal initialization
- ``'uniform'`` — random uniform
- ``'zeros'`` — zero initialization
- ``'ones'`` — ones initialization
- ``'fill_value'`` — constant fill value (pass ``value=`` kwarg)

For unsupervised models, ``selection_init`` and ``mean_init`` from
``prosemble.core.initializers`` provide classless alternatives.

You can also pass a callable for custom initialization (e.g., ``literal_init(values)``
from ``prosemble.core.initializers``).

GRLVQ — Feature Relevances
---------------------------

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
-------------------------------

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
-----------------------

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
-------------------------

Learns per-prototype invariance subspaces. The tangent distance measures
distance only in directions orthogonal to the invariance subspace:
:math:`d(x, w_k) = \|(I - \Omega_k \Omega_k^T)(x - w_k)\|^2`.

.. code-block:: python

   from prosemble.models import GTLVQ

   model = GTLVQ(
       n_prototypes_per_class=2,
       tangent_dim=1,        # 1D invariance subspace
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)

CELVQ — Cross-Entropy
----------------------

Replaces the GLVQ :math:`\mu` cost with cross-entropy loss over class
probabilities derived from distances. Produces calibrated probabilities.

.. code-block:: python

   from prosemble.models import CELVQ

   model = CELVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)
   proba = model.predict_proba(X_test)  # calibrated class probabilities

LVQ1 and LVQ2.1 — Classic Non-Gradient
---------------------------------------

Simple competitive learning without explicit gradients. LVQ1 updates only
the nearest prototype; LVQ2.1 updates both nearest correct and incorrect.

.. code-block:: python

   from prosemble.models import LVQ1, LVQ21

   model = LVQ1(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

Probabilistic Models (SLVQ, RSLVQ)
-----------------------------------

Treat prototypes as Gaussian components. RSLVQ provides robust
log-likelihood training with confidence-based rejection.

.. code-block:: python

   from prosemble.models import RSLVQ

   model = RSLVQ(
       n_prototypes_per_class=2,
       sigma=1.0,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

Median LVQ
----------

Constrains prototypes to be actual data points for maximum interpretability.

.. code-block:: python

   from prosemble.models import MedianLVQ

   model = MedianLVQ(
       n_prototypes_per_class=1,
       max_iter=50,
       random_seed=42,
   )
   model.fit(X_train, y_train)
   # model.prototypes_ are actual training examples

Deep Variants (LVQMLN, PLVQ)
-----------------------------

Add a trainable MLP backbone for nonlinear feature extraction.
Prototypes live in the latent space (not the input space).

.. code-block:: python

   from prosemble.models import LVQMLN

   model = LVQMLN(
       n_prototypes_per_class=2,
       hidden_dims=[64, 32],  # MLP architecture
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)

Siamese Variants
----------------

Like LVQMLN, but prototypes are in the **input space** and pass through
the same backbone as input data. This makes prototypes interpretable.

.. code-block:: python

   from prosemble.models import SiameseGLVQ, SiameseGMLVQ, SiameseGTLVQ

   model = SiameseGLVQ(
       n_prototypes_per_class=2,
       hidden_dims=[64, 32],
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)
   # model.prototypes_ is in the original input space

Image LVQ
---------

Siamese architecture with a CNN backbone for image classification.

.. code-block:: python

   from prosemble.models import ImageGLVQ, ImageGMLVQ, ImageGTLVQ

   model = ImageGLVQ(
       n_prototypes_per_class=2,
       max_iter=50,
       lr=0.001,
   )
   # X_train should be image data: (n_samples, height, width, channels)
   model.fit(X_train, y_train)

CBC — Classification-By-Components
-----------------------------------

Uses classless components with learned reasoning matrices for explainable
classification.

.. code-block:: python

   from prosemble.models import CBC

   model = CBC(
       n_components=6,
       num_classes=3,
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)

Supervised Relevance Neural Gas (SRNG)
--------------------------------------

Combines GLVQ loss with Neural Gas neighborhood cooperation. All same-class
prototypes are updated per sample, weighted by rank. SRNG also learns
per-feature relevance weights (like GRLVQ).

.. code-block:: python

   from prosemble.models import SRNG

   model = SRNG(
       n_prototypes_per_class=3,
       lambda_init=5.0,       # initial neighborhood range
       lambda_final=0.01,     # final (narrower)
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

   # Inspect relevances (SRNG learns feature weights like GRLVQ)
   print(model.relevances_)

Cross-Entropy Neural Gas (CELVQ-NG Family)
-------------------------------------------

The CELVQ-NG family combines cross-entropy loss over all-class softmax logits
with Neural Gas rank-based neighborhood cooperation. Unlike SRNG (which uses
pairwise GLVQ :math:`\mu` cost), CELVQ-NG considers all classes simultaneously
via softmax, providing better calibrated probabilities and gradient flow to
all prototypes.

Neural Gas cooperation replaces the hard per-class ``min`` pooling in CELVQ
with NG-weighted soft pooling: for each class, prototypes are ranked by
distance and weighted by :math:`h_k = \exp(-\text{rank} / \gamma)`. When
:math:`\gamma \to 0`, CELVQ-NG recovers standard CELVQ.

**CELVQ_NG** — Euclidean distance (base variant):

.. code-block:: python

   from prosemble.models import CELVQ_NG

   model = CELVQ_NG(
       n_prototypes_per_class=3,
       gamma_init=5.0,        # initial neighborhood range
       gamma_final=0.01,      # final (narrower)
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)
   proba = model.predict_proba(X_test)  # calibrated probabilities

**MCELVQ_NG** — Global Omega matrix metric learning:

.. code-block:: python

   from prosemble.models import MCELVQ_NG

   model = MCELVQ_NG(
       n_prototypes_per_class=3,
       latent_dim=2,          # project to 2D
       gamma_init=5.0,
       gamma_final=0.01,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

   # Learned metric matrices
   print(model.omega_matrix.shape)   # (n_features, latent_dim)
   print(model.lambda_matrix)        # Omega^T @ Omega — feature importance

**LCELVQ_NG** — Per-prototype local Omega matrices:

.. code-block:: python

   from prosemble.models import LCELVQ_NG

   model = LCELVQ_NG(
       n_prototypes_per_class=2,
       latent_dim=2,
       gamma_init=5.0,
       gamma_final=0.01,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

   # Each prototype has its own Omega_k
   print(model.omegas_.shape)  # (n_prototypes, n_features, latent_dim)

**TCELVQ_NG** — Tangent subspace distance:

.. code-block:: python

   from prosemble.models import TCELVQ_NG

   model = TCELVQ_NG(
       n_prototypes_per_class=2,
       subspace_dim=1,        # 1D invariance subspace
       gamma_init=5.0,
       gamma_final=0.01,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

   # Learned orthogonal tangent bases
   print(model.omegas_.shape)  # (n_prototypes, n_features, subspace_dim)

The tangent variant measures distance orthogonal to learned invariance
subspaces: :math:`d(x, w_k) = \|(I - \Omega_k \Omega_k^T)(x - w_k)\|^2`.
Best suited for high-dimensional data with invariance structure (images,
spectra, signals).

.. list-table:: CELVQ-NG Family Summary
   :header-rows: 1
   :widths: 20 30 25 25

   * - Model
     - Distance Metric
     - Learnable Parameters
     - Best For
   * - CELVQ_NG
     - Euclidean
     - Prototypes only
     - General-purpose, fast training
   * - MCELVQ_NG
     - :math:`\|\Omega(x-w)\|^2`
     - Global :math:`\Omega` matrix
     - Feature selection, dimensionality reduction
   * - LCELVQ_NG
     - :math:`\|\Omega_k(x-w_k)\|^2`
     - Per-prototype :math:`\Omega_k`
     - Heterogeneous feature spaces
   * - TCELVQ_NG
     - :math:`\|(I-\Omega_k\Omega_k^T)(x-w_k)\|^2`
     - Tangent bases :math:`\Omega_k`
     - High-dimensional data with invariances

Common Patterns
---------------

**Resume training:**

.. code-block:: python

   model.fit(X_train, y_train, max_iter=50)
   model.fit(X_train, y_train, resume=True, max_iter=50)  # continue from last state

**Fitted attributes** (available after ``fit``):

- ``model.prototypes_`` — prototype positions
- ``model.prototype_labels_`` — class labels per prototype
- ``model.n_iter_`` — number of iterations run
- ``model.loss_`` — final loss value
- ``model.loss_history_`` — loss per iteration

**All models** support:

- ``predict(X)`` — hard labels
- ``predict_proba(X)`` — soft class probabilities
- ``save(path)`` / ``Model.load(path)`` — persistence
