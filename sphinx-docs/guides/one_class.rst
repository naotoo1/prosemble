One-Class Classification
========================

Prosemble provides 21 one-class classification models across three families,
all implemented in JAX with JIT compilation and ``lax.scan`` training loops.
Every model follows the same ``fit`` / ``predict`` / ``decision_function`` /
``predict_with_reject`` API.

All one-class models learn per-prototype **visibility thresholds**
:math:`\theta_k` that define the boundary between target (normal) and
non-target (outlier) samples.

OC-GLVQ — Baseline
-------------------

One-Class Generalized Learning Vector Quantization. Replaces the standard
GLVQ competing distance :math:`d^-` with learned per-prototype visibility
thresholds :math:`\theta_k`. Only the nearest prototype contributes to
the loss.

.. code-block:: python

   import jax.numpy as jnp
   from prosemble.datasets import load_iris_jax
   from prosemble.models import OCGLVQ

   # Load data — treat class 0 as target, rest as outliers
   dataset = load_iris_jax()
   X, y_raw = dataset.input_data, dataset.labels
   y = jnp.where(y_raw == 0, 0, 1).astype(jnp.int32)

   model = OCGLVQ(
       n_prototypes=3,
       target_label=0,          # target (normal) class label
       beta=10.0,               # sigmoid steepness
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X, y)

   # Hard labels
   preds = model.predict(X)

   # Decision scores in [0, 1] — higher = more target-like
   scores = model.decision_function(X)

   # Predict with reject option
   preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
   # Returns: 0 (target), 1 (outlier), -1 (uncertain/rejected)

   # Learned thresholds
   print(model.thetas_)           # (n_prototypes,)
   print(model.visibility_radii)  # same as thetas_

OCGRLVQ — Feature Relevances
-----------------------------

Adds per-feature relevance weights :math:`\lambda_j` that reveal which
features matter for target-vs-outlier separation.

.. code-block:: python

   from prosemble.models import OCGRLVQ

   model = OCGRLVQ(
       n_prototypes=3,
       target_label=0,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X, y)

   # Inspect learned relevances
   print(model.relevances_)  # shape: (n_features,), sums to 1

OCGMLVQ — Matrix Metric Learning
---------------------------------

Learns a global linear transformation :math:`\Omega` such that distance
is :math:`d(x, w) = \|\Omega(x - w)\|^2`. Captures feature correlations.

.. code-block:: python

   from prosemble.models import OCGMLVQ

   model = OCGMLVQ(
       n_prototypes=3,
       latent_dim=2,            # project to 2D
       target_label=0,
       max_iter=100,
       lr=0.001,
   )
   model.fit(X, y)

   print(model.omega_matrix.shape)   # (n_features, latent_dim)
   print(model.lambda_matrix.shape)  # Omega^T @ Omega

OCLGMLVQ — Local Metrics
-------------------------

Each prototype gets its own :math:`\Omega_k` matrix, allowing different
regions of the feature space to use different metrics.

.. code-block:: python

   from prosemble.models import OCLGMLVQ

   model = OCLGMLVQ(
       n_prototypes=3,
       latent_dim=2,
       target_label=0,
       max_iter=100,
       lr=0.001,
   )
   model.fit(X, y)

   print(model.omegas_.shape)  # (n_prototypes, n_features, latent_dim)

OCGTLVQ — Tangent Distance
---------------------------

Learns per-prototype invariance subspaces. The tangent distance measures
distance only in directions orthogonal to the invariance subspace:
:math:`d(x, w_k) = \|(I - \Omega_k \Omega_k^T)(x - w_k)\|^2`.

.. code-block:: python

   from prosemble.models import OCGTLVQ

   model = OCGTLVQ(
       n_prototypes=3,
       subspace_dim=2,          # tangent subspace dimension
       target_label=0,
       max_iter=100,
       lr=0.001,
   )
   model.fit(X, y)

OC-GLVQ with Neural Gas
^^^^^^^^^^^^^^^^^^^^^^^^

Neural Gas variants replace hard nearest-prototype selection with
rank-based neighborhood cooperation. All prototypes contribute, weighted
by :math:`h_k = \exp(-\text{rank}_k / \gamma)`. Gamma decays from broad
to sharp during training.

.. code-block:: python

   from prosemble.models import OCGLVQ_NG

   model = OCGLVQ_NG(
       n_prototypes=3,
       target_label=0,
       gamma_init=5.0,          # initial neighborhood range
       gamma_final=0.01,        # final (narrower)
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X, y)

   print(f"Final gamma: {model.gamma_:.4f}")

All five OC-GLVQ metric variants have NG counterparts:

.. list-table:: OC-GLVQ Neural Gas Variants
   :header-rows: 1
   :widths: 30 30 40

   * - Model
     - Base
     - Metric
   * - ``OCGLVQ_NG``
     - OCGLVQ
     - Euclidean
   * - ``OCGRLVQ_NG``
     - OCGRLVQ
     - Per-feature relevance
   * - ``OCGMLVQ_NG``
     - OCGMLVQ
     - Global :math:`\Omega`
   * - ``OCLGMLVQ_NG``
     - OCLGMLVQ
     - Per-prototype :math:`\Omega_k`
   * - ``OCGTLVQ_NG``
     - OCGTLVQ
     - Tangent subspace

OC-RSLVQ — Probabilistic One-Class
-----------------------------------

One-Class Robust Soft LVQ. Replaces hard nearest-prototype selection with
soft Gaussian mixture responsibilities. All prototypes contribute,
weighted by proximity: :math:`p(k|x) = \text{softmax}(-d_k / 2\sigma^2)`.

.. code-block:: python

   from prosemble.models import OCRSLVQ

   model = OCRSLVQ(
       sigma=1.0,               # Gaussian bandwidth
       n_prototypes=3,
       target_label=0,
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X, y)

   # Decision scores
   scores = model.decision_function(X)
   mean_target = float(jnp.mean(scores[y == 0]))
   mean_outlier = float(jnp.mean(scores[y == 1]))
   print(f"Mean target score:  {mean_target:.4f}")
   print(f"Mean outlier score: {mean_outlier:.4f}")

   # Predict with reject option
   preds = model.predict_with_reject(X, upper=0.7, lower=0.3)

OCMRSLVQ — Matrix Metric
-------------------------

Combines OC-RSLVQ's soft Gaussian weighting with a global :math:`\Omega`
projection matrix.

.. code-block:: python

   from prosemble.models import OCMRSLVQ

   model = OCMRSLVQ(
       sigma=1.0,
       n_prototypes=3,
       target_label=0,
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X, y)

   print(f"Omega shape: {model.omega_matrix.shape}")
   print(f"Lambda (relevance matrix):\n{model.lambda_matrix}")

OCLMRSLVQ — Local Matrix Metric
--------------------------------

Per-prototype :math:`\Omega_k` matrices with soft Gaussian weighting.

.. code-block:: python

   from prosemble.models import OCLMRSLVQ

   model = OCLMRSLVQ(
       sigma=1.0,
       n_prototypes=3,
       latent_dim=2,
       target_label=0,
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X, y)

   print(model.omegas_.shape)  # (n_prototypes, n_features, latent_dim)

OC-RSLVQ with Neural Gas
^^^^^^^^^^^^^^^^^^^^^^^^^

Combines soft Gaussian mixture with Neural Gas rank-based cooperation.
Prototype weights are the product of Gaussian responsibility and NG rank:
:math:`w_k = p(k|x) \cdot h_k / \sum_j p(j|x) \cdot h_j`.

.. code-block:: python

   from prosemble.models import OCRSLVQ_NG

   model = OCRSLVQ_NG(
       sigma=1.0,
       n_prototypes=3,
       target_label=0,
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X, y)

   print(f"Final loss: {model.loss_:.4f}")
   print(f"Final gamma: {model.gamma_:.4f}")

.. list-table:: OC-RSLVQ Neural Gas Variants
   :header-rows: 1
   :widths: 30 30 40

   * - Model
     - Base
     - Metric
   * - ``OCRSLVQ_NG``
     - OCRSLVQ
     - Euclidean
   * - ``OCMRSLVQ_NG``
     - OCMRSLVQ
     - Global :math:`\Omega`
   * - ``OCLMRSLVQ_NG``
     - OCLMRSLVQ
     - Per-prototype :math:`\Omega_k`

Matrix NG variants add ``latent_dim``:

.. code-block:: python

   from prosemble.models import OCMRSLVQ_NG, OCLMRSLVQ_NG

   # Global omega + Gaussian + NG
   model = OCMRSLVQ_NG(
       sigma=1.0,
       latent_dim=2,
       n_prototypes=3,
       target_label=0,
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X, y)

   print(f"Omega shape: {model.omega_matrix.shape}")
   print(f"Final gamma: {model.gamma_:.4f}")

SVQ-OCC — Dual-Objective One-Class
-----------------------------------

Supervised Vector Quantization One-Class Classification. Balances two
objectives:

- **R cost**: Neural Gas representation learning on target data
- **C cost**: Classification cost via per-prototype responsibilities

Total loss: :math:`E = \alpha \cdot R + (1 - \alpha) \cdot C`

.. code-block:: python

   from prosemble.models import SVQOCC

   model = SVQOCC(
       n_prototypes=3,
       target_label=0,
       alpha=0.5,               # balance: R vs C cost
       cost_function='contrastive',
       response_type='gaussian',
       sigma=0.1,               # Heaviside sigmoid sharpness
       gamma_resp=1.0,          # response bandwidth
       lambda_init=5.0,         # initial NG lambda
       lambda_final=0.01,       # final NG lambda
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X, y)

   scores = model.decision_function(X)
   preds = model.predict(X)

**Cost functions** (``cost_function``):

- ``'contrastive'`` — contrastive score (default)
- ``'brier'`` — Brier score
- ``'cross_entropy'`` — cross-entropy loss

**Response types** (``response_type``):

- ``'gaussian'`` — :math:`p_k = \text{softmax}(-\gamma \cdot d_k)` (default)
- ``'student_t'`` — :math:`p_k \propto (1 + d_k / \nu)^{-(\nu+1)/2}`
- ``'uniform'`` — :math:`p_k = 1/K`

SVQ-OCC Metric Variants
^^^^^^^^^^^^^^^^^^^^^^^^

Like the OC-GLVQ family, SVQ-OCC has metric-adapted variants:

.. list-table:: SVQ-OCC Variants
   :header-rows: 1
   :widths: 25 35 40

   * - Model
     - Metric
     - Key Parameter
   * - ``SVQOCC``
     - Euclidean
     - —
   * - ``SVQOCC_R``
     - Per-feature relevance
     - ``relevances_``
   * - ``SVQOCC_M``
     - Global :math:`\Omega`
     - ``latent_dim``
   * - ``SVQOCC_LM``
     - Per-prototype :math:`\Omega_k`
     - ``latent_dim``
   * - ``SVQOCC_T``
     - Tangent subspace
     - ``subspace_dim``

.. code-block:: python

   from prosemble.models import SVQOCC_R, SVQOCC_M, SVQOCC_LM, SVQOCC_T

   # Relevance variant
   model = SVQOCC_R(n_prototypes=3, target_label=0, max_iter=100, lr=0.01)
   model.fit(X, y)
   print(model.relevances_)      # (n_features,)

   # Matrix variant
   model = SVQOCC_M(n_prototypes=3, target_label=0, latent_dim=2,
                     max_iter=100, lr=0.01)
   model.fit(X, y)
   print(model.omega_matrix.shape)

Common Patterns
---------------

**Data preparation** — one-class models expect binary labels (target=0,
outlier=1):

.. code-block:: python

   import jax.numpy as jnp
   from prosemble.datasets import load_iris_jax

   dataset = load_iris_jax()
   X, y_raw = dataset.input_data, dataset.labels
   y = jnp.where(y_raw == 0, 0, 1).astype(jnp.int32)

**Decision function** — all models return scores in [0, 1]:

.. code-block:: python

   scores = model.decision_function(X)
   # Higher score = more target-like
   # Lower score = more outlier-like

**Reject option** — three-way classification:

.. code-block:: python

   preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
   # 0: accepted (target)
   # 1: rejected (outlier)
   # -1: uncertain (between lower and upper)

**Resume training:**

.. code-block:: python

   model.fit(X, y, max_iter=50)
   model.fit(X, y, resume=True)  # continue from last state

**Fitted attributes** (after ``fit``):

- ``model.prototypes_`` — prototype positions
- ``model.thetas_`` — per-prototype visibility thresholds
- ``model.n_iter_`` — number of iterations run
- ``model.loss_`` — final loss value
- ``model.loss_history_`` — loss per iteration
- ``model.visibility_radii`` — same as ``thetas_``

**NG-specific attributes** (OC-*-NG and SVQ-OCC models):

- ``model.gamma_`` — final gamma / lambda value after training

**Matrix-specific attributes** (M / LM variants):

- ``model.omega_matrix`` — learned :math:`\Omega` projection
- ``model.lambda_matrix`` — :math:`\Omega^T \Omega` (feature importance)
- ``model.omegas_`` — per-prototype :math:`\Omega_k` (local variants)
