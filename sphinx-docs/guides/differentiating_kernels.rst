Differentiating Kernel Models
==============================

Differentiating kernel models replace Euclidean distances with **kernel-induced
distances** in prototype-based learning. The kernel parameters are adapted via
gradient descent alongside the prototypes, enabling the model to learn an
optimal non-linear similarity measure from data.

Mathematical Background
-----------------------

Gaussian Kernel Distance
^^^^^^^^^^^^^^^^^^^^^^^^

For a Gaussian kernel with bandwidth :math:`\sigma`:

.. math::

   \kappa(x, w) = \exp\!\left(-\frac{\|x - w\|^2}{2\sigma^2}\right)

the induced distance in feature space is:

.. math::

   d_\kappa^2(x, w) = \|\phi(x) - \phi(w)\|^2
                     = 2\bigl(1 - \kappa(x, w)\bigr)
                     = 2\left(1 - \exp\!\left(
                         -\frac{\|x - w\|^2}{2\sigma^2}
                       \right)\right)

This distance is bounded in :math:`[0, 2]` regardless of input magnitude,
making it naturally robust to outliers.

Relevance-Weighted Kernel Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adding per-feature relevance weights :math:`\lambda_j = \text{softmax}(\text{relevances})_j`:

.. math::

   d_\kappa^2(x, w_k) = 2\left(1 - \exp\!\left(
       -\frac{\sum_j \lambda_j (x_j - w_{kj})^2}{2\sigma_k^2}
   \right)\right)

This combines feature selection with kernel distance, identifying which
input dimensions are most important for classification.

Exponential Kernel Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The exponential kernel uses a learned transformation matrix
:math:`\hat\Lambda = \hat\Omega \hat\Omega^T`:

.. math::

   \kappa_{\exp}(x, w) = \exp\!\bigl(x^T \hat\Lambda\, w\bigr)

Unlike the Gaussian kernel, :math:`\kappa_{\exp}(v, v) \neq 1`, so the full
three-term distance formula is required:

.. math::

   d_\kappa^2(x, w) = \exp\!\bigl(x^T \hat\Lambda\, x\bigr)
                    + \exp\!\bigl(w^T \hat\Lambda\, w\bigr)
                    - 2\exp\!\bigl(x^T \hat\Lambda\, w\bigr)


Supervised Models
-----------------

DKGLVQ
^^^^^^

Differentiating Kernel GLVQ. Each prototype :math:`w_k` has a learnable
bandwidth :math:`\sigma_k` adapted via gradient descent.

.. code-block:: python

   from prosemble.models import DKGLVQ
   from prosemble.datasets import load_iris_jax

   dataset = load_iris_jax()
   X, y = dataset.input_data, dataset.target

   model = DKGLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       sigma_init='median',   # per-class median distance initialization
       sigma_min=1e-3,        # prevent bandwidth collapse
       random_seed=42,
   )
   model.fit(X, y)

   preds = model.predict(X)
   print(f"Accuracy: {(preds == y).mean():.2%}")
   print(f"Learned bandwidths: {model.kernel_bandwidths}")

The ``sigma_init`` parameter controls initialization:

- ``'median'`` (default): per-class median distance from prototype to class members
- ``'mean'``: per-class mean distance
- ``float``: fixed value for all prototypes

DKGRLVQ
^^^^^^^^

Differentiating Kernel GRLVQ. Combines per-feature relevance weighting
with per-prototype kernel bandwidth adaptation.

.. code-block:: python

   from prosemble.models import DKGRLVQ

   model = DKGRLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       sigma_init='median',
       sigma_min=1e-3,
       random_seed=42,
   )
   model.fit(X, y)

   preds = model.predict(X)
   print(f"Accuracy: {(preds == y).mean():.2%}")
   print(f"Relevance profile: {model.relevance_profile}")
   print(f"Learned bandwidths: {model.kernel_bandwidths}")

The ``relevance_profile`` property returns the normalized feature relevance
weights :math:`\lambda = \text{softmax}(\text{relevances})`, identifying
which features are most discriminative.

DKGMLVQ
^^^^^^^^

Differentiating Kernel GMLVQ with the exponential kernel. Learns a global
transformation matrix :math:`\hat\Omega` of shape ``(d, latent_dim)``.

.. code-block:: python

   from prosemble.models import DKGMLVQ

   model = DKGMLVQ(
       n_prototypes_per_class=2,
       max_iter=200,
       lr=0.01,
       latent_dim=None,          # defaults to input dim
       omega_hat_scale=0.1,      # small init prevents exp overflow
       random_seed=42,
   )
   model.fit(X, y)

   preds = model.predict(X)
   print(f"Omega hat shape: {model.omega_hat_matrix.shape}")
   print(f"Lambda hat shape: {model.lambda_hat_matrix.shape}")

The ``lambda_hat_matrix`` property returns the symmetric positive
semi-definite matrix :math:`\hat\Lambda = \hat\Omega \hat\Omega^T`,
which can be analyzed for feature correlations learned by the model.


One-Class Models
-----------------

The one-class differentiating kernel models combine OC-GLVQ's
:math:`\theta`-based hypothesis testing with kernel distances. In standard
OC-GLVQ, the classifier function is:

.. math::

   \mu_{k^*}(x_i) = s_i \cdot \frac{d(x_i, w_{k^*}) - \theta_{k^*}}{d(x_i, w_{k^*}) + \theta_{k^*}}

where :math:`k^*` is the nearest prototype, :math:`\theta_{k^*}` is a learned
per-prototype visibility threshold, and :math:`s_i = +1` for target,
:math:`-1` for outlier. The OC-DK variants replace the Euclidean distance
:math:`d` with kernel distances.

**Critical design detail:** The :math:`\theta_k` thresholds are initialized
in *kernel distance scale*, not Euclidean scale. Gaussian kernel distances are
bounded in :math:`[0, 2]`, so Euclidean-initialized thetas would be far too
large.

OCDKGLVQ
^^^^^^^^^

One-class classification with Gaussian kernel distance and per-prototype
bandwidth adaptation.

.. code-block:: python

   from prosemble.models import OCDKGLVQ
   import jax
   import jax.numpy as jnp

   # Generate one-class dataset
   key = jax.random.PRNGKey(42)
   k1, k2 = jax.random.split(key)
   X_target = jax.random.normal(k1, (100, 4)) * 0.5
   X_outlier = jax.random.normal(k2, (30, 4)) * 0.5 + 3.0
   X = jnp.concatenate([X_target, X_outlier])
   y = jnp.concatenate([jnp.zeros(100, dtype=jnp.int32),
                        jnp.ones(30, dtype=jnp.int32)])

   model = OCDKGLVQ(
       n_prototypes=3,
       max_iter=100,
       lr=0.01,
       sigma_init='median',
       sigma_min=1e-3,
       target_label=0,
       random_seed=42,
   )
   model.fit(X, y)

   scores = model.decision_function(X)
   preds = model.predict(X)
   print(f"Learned bandwidths: {model.kernel_bandwidths}")
   print(f"Visibility radii: {model.visibility_radii}")

OCDKGRLVQ
^^^^^^^^^^

One-class classification with relevance-weighted kernel distance,
per-prototype bandwidth, and per-feature relevance learning.

.. code-block:: python

   from prosemble.models import OCDKGRLVQ

   model = OCDKGRLVQ(
       n_prototypes=3,
       max_iter=100,
       lr=0.01,
       sigma_init='median',
       sigma_min=1e-3,
       target_label=0,
       random_seed=42,
   )
   model.fit(X, y)

   scores = model.decision_function(X)
   print(f"Relevance profile: {model.relevance_profile}")
   print(f"Learned bandwidths: {model.kernel_bandwidths}")

The ``relevance_profile`` property returns the softmax-normalized per-feature
weights, identifying which features are most important for the one-class
boundary.

OCDKGMLVQ
^^^^^^^^^^

One-class classification with exponential kernel distance and a learned
transformation matrix :math:`\hat\Omega`.

.. code-block:: python

   from prosemble.models import OCDKGMLVQ

   model = OCDKGMLVQ(
       n_prototypes=3,
       max_iter=100,
       lr=0.01,
       latent_dim=None,
       omega_hat_scale=0.1,
       target_label=0,
       random_seed=42,
   )
   model.fit(X, y)

   scores = model.decision_function(X)
   print(f"Omega hat shape: {model.omega_hat_matrix.shape}")
   print(f"Lambda hat (PSD): {model.lambda_hat_matrix.shape}")

Supervised Models with Neural Gas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The supervised DK-NG variants combine differentiating kernel distances with
Neural Gas class-aware neighborhood cooperation. All **same-class** prototypes
participate in the loss, weighted by their distance rank:

.. math::

   h_k = \exp\left(-\frac{\text{rank}_k}{\gamma}\right), \quad
   \text{only for } \text{label}(w_k) = \text{label}(x)

where :math:`\gamma` decays during training from ``gamma_init`` to
``gamma_final``, and the GLVQ margin is computed per prototype:

.. math::

   \mu_k = \frac{d_\kappa(x, w_k) - d^-}{d_\kappa(x, w_k) + d^-}

with :math:`d^-` being the nearest different-class prototype distance.

.. code-block:: python

   from prosemble.models import DKGLVQ_NG

   model = DKGLVQ_NG(
       n_prototypes_per_class=3,
       max_iter=100,
       lr=0.01,
       sigma_init='median',
       gamma_init=1.5,
       gamma_final=0.01,
       random_seed=42,
   )
   model.fit(X, y)
   preds = model.predict(X)
   print(f"Final gamma: {model.gamma_}")

The relevance-weighted variant (``DKGRLVQ_NG``) adds per-feature relevance
weights, while the matrix variant (``DKGMLVQ_NG``) uses exponential kernel
distance with learnable :math:`\hat\Omega` transformation.


One-Class Models with Neural Gas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The OC-DK-NG variants extend the one-class kernel models with Neural Gas
neighborhood cooperation. Instead of only the nearest prototype contributing
to the loss, **all** prototypes participate weighted by their distance rank:

.. math::

   h_k = \exp\left(-\frac{\text{rank}_k}{\gamma}\right)

where :math:`\gamma` decays during training from ``gamma_init`` to
``gamma_final``.

.. code-block:: python

   from prosemble.models import OCDKGLVQ_NG

   model = OCDKGLVQ_NG(
       n_prototypes=3,
       max_iter=100,
       lr=0.01,
       sigma_init='median',
       gamma_init=1.5,
       gamma_final=0.01,
       target_label=0,
       random_seed=42,
   )
   model.fit(X, y)
   print(f"Final gamma: {model.gamma_}")

The relevance-weighted variant (``OCDKGRLVQ_NG``) and matrix variant
(``OCDKGMLVQ_NG``) follow the same pattern, combining their respective
kernel distances with NG cooperation.


Unsupervised Models
-------------------

The unsupervised kernel models use the Gaussian kernel distance for prototype
ranking and BMU selection, but :math:`\sigma` is a **fixed hyperparameter**
(not learned). Prototypes live in the original data space — only the distance
metric changes.

DKNeuralGas
^^^^^^^^^^^^

Neural Gas with Gaussian kernel distance for ranking.

.. code-block:: python

   from prosemble.models import DKNeuralGas
   from prosemble.datasets import load_iris_jax

   dataset = load_iris_jax()
   X = dataset.input_data

   model = DKNeuralGas(
       n_prototypes=10,
       kernel_sigma=1.0,
       max_iter=100,
       lr_init=0.5,
       lr_final=0.01,
       lambda_final=0.01,
       random_seed=42,
   )
   model.fit(X)

   labels = model.predict(X)
   print(f"Energy: {model.loss_:.4f}")

DKKohonenSOM
^^^^^^^^^^^^^

Kohonen SOM with Gaussian kernel distance for BMU selection. The grid
neighborhood is unchanged — only the data-space metric changes.

.. code-block:: python

   from prosemble.models import DKKohonenSOM

   model = DKKohonenSOM(
       grid_height=5,
       grid_width=5,
       kernel_sigma=1.0,
       sigma_init=2.0,
       sigma_final=0.5,
       lr_init=0.5,
       lr_final=0.01,
       max_iter=100,
       random_seed=42,
   )
   model.fit(X)

   bmu_coords = model.bmu_map(X)
   print(f"BMU coordinates shape: {bmu_coords.shape}")

DKHeskesSOM
^^^^^^^^^^^^

Heskes SOM with Gaussian kernel distance. The Heskes BMU criterion selects
the unit whose **entire neighborhood** best represents the sample:

.. math::

   c^*(x) = \arg\min_c \sum_k h(k, c) \cdot d_\kappa^2(x, w_k)

.. code-block:: python

   from prosemble.models import DKHeskesSOM

   model = DKHeskesSOM(
       grid_height=5,
       grid_width=5,
       kernel_sigma=1.0,
       sigma_init=2.0,
       sigma_final=0.5,
       max_iter=100,
       random_seed=42,
   )
   model.fit(X)

   bmu_coords = model.bmu_map(X)
   print(f"Energy: {model.loss_:.4f}")


Choosing a Model
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Model
     - Kernel
     - Learned Params
     - Best For
   * - DKGLVQ
     - Gaussian
     - :math:`w_k, \sigma_k`
     - Per-prototype bandwidth adaptation
   * - DKGRLVQ
     - Gaussian (weighted)
     - :math:`w_k, \sigma_k, \lambda`
     - Feature selection + kernel adaptation
   * - DKGMLVQ
     - Exponential
     - :math:`w_k, \hat\Omega`
     - Full metric adaptation in kernel space
   * - DKGLVQ_NG
     - Gaussian
     - :math:`w_k, \sigma_k, \gamma`
     - Supervised kernel + NG cooperation
   * - DKGRLVQ_NG
     - Gaussian (weighted)
     - :math:`w_k, \sigma_k, \lambda, \gamma`
     - Supervised kernel + relevances + NG cooperation
   * - DKGMLVQ_NG
     - Exponential
     - :math:`w_k, \hat\Omega, \gamma`
     - Supervised kernel matrix + NG cooperation
   * - OCDKGLVQ
     - Gaussian
     - :math:`w_k, \sigma_k, \theta_k`
     - One-class with kernel bandwidth adaptation
   * - OCDKGRLVQ
     - Gaussian (weighted)
     - :math:`w_k, \sigma_k, \lambda, \theta_k`
     - One-class with feature selection + kernel
   * - OCDKGMLVQ
     - Exponential
     - :math:`w_k, \hat\Omega, \theta_k`
     - One-class with full metric adaptation in kernel space
   * - OCDKGLVQ_NG
     - Gaussian
     - :math:`w_k, \sigma_k, \theta_k, \gamma`
     - One-class kernel + NG cooperation
   * - OCDKGRLVQ_NG
     - Gaussian (weighted)
     - :math:`w_k, \sigma_k, \lambda, \theta_k, \gamma`
     - One-class kernel + relevances + NG cooperation
   * - OCDKGMLVQ_NG
     - Exponential
     - :math:`w_k, \hat\Omega, \theta_k, \gamma`
     - One-class kernel matrix + NG cooperation
   * - DKNeuralGas
     - Gaussian (fixed :math:`\sigma`)
     - :math:`w_k`
     - Unsupervised clustering with kernel distance
   * - DKKohonenSOM
     - Gaussian (fixed :math:`\sigma`)
     - :math:`w_k`
     - SOM visualization with kernel distance
   * - DKHeskesSOM
     - Gaussian (fixed :math:`\sigma`)
     - :math:`w_k`
     - Principled SOM with kernel distance

Riemannian Variants
-------------------

Differentiating kernel distances can also be applied on Riemannian manifolds.
Three models combine the RiemannianSRNG framework (prototypes on manifold,
NG rank cooperation, manifold projection) with kernel distance formulas:

- **RiemannianDKGLVQ** — Gaussian kernel on geodesic distance:
  :math:`d_\kappa^2(x, w_k) = 2(1 - \exp(-d_{\text{geo}}^2(x, w_k) / 2\sigma_k^2))`
- **RiemannianDKGRLVQ** — Relevance-weighted kernel in tangent space:
  :math:`d_\kappa^2(x, w_k) = 2(1 - \exp(-\sum_j \lambda_j v_j^2 / 2\sigma_k^2))`
  where :math:`v = \text{Log}_{w_k}(x)_{\text{flat}}`
- **RiemannianDKGMLVQ** — Exponential kernel in tangent space:
  :math:`d_\kappa^2(x, w_k) = \exp(v^T \hat\Lambda v) - 1`
  where :math:`\hat\Lambda = \hat\Omega \hat\Omega^T`

All three support SO(n), SPD(n), and Grassmannian(n,k) manifolds.

.. code-block:: python

   from prosemble.core.manifolds import SO
   from prosemble.models import RiemannianDKGLVQ

   manifold = SO(3)
   model = RiemannianDKGLVQ(
       manifold=manifold, n_prototypes_per_class=2,
       max_iter=100, lr=0.01, use_scan=False,
   )
   model.fit(X_train, y_train)
   preds = model.predict(X_test)

   # Inspect learned bandwidths
   print(model.kernel_bandwidths)

Riemannian Metric-Adapted DK Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nine additional models combine metric adaptation (global/local/subspace) with
kernel distance learning on Riemannian manifolds, completing a 3x3 grid
(Gaussian/Relevance/Matrix kernels x SMNG/SLNG/STNG bases):

**Gaussian kernel variants** (per-prototype :math:`\sigma_k`):

- **RiemannianDKSMNG** — :math:`d_\kappa^2 = 2(1 - \exp(-\|\Omega \cdot v\|^2 / 2\sigma_k^2))`
- **RiemannianDKSLNG** — :math:`d_\kappa^2 = 2(1 - \exp(-\|\Omega_k \cdot v\|^2 / 2\sigma_k^2))`
- **RiemannianDKSTNG** — :math:`d_\kappa^2 = 2(1 - \exp(-\|(I - \Omega_k\Omega_k^T) \cdot v\|^2 / 2\sigma_k^2))`

**Relevance kernel variants** (:math:`\sigma_k` + relevance :math:`\lambda`):

- **RiemannianDKRSMNG** — :math:`d_\kappa^2 = 2(1 - \exp(-\sum_j \lambda_j (\Omega \cdot v)_j^2 / 2\sigma_k^2))`
- **RiemannianDKRSLNG** — :math:`d_\kappa^2 = 2(1 - \exp(-\sum_j \lambda_j (\Omega_k \cdot v)_j^2 / 2\sigma_k^2))`
- **RiemannianDKRSTNG** — :math:`d_\kappa^2 = 2(1 - \exp(-\sum_j \lambda_j r_j^2 / 2\sigma_k^2))`

**Matrix kernel variants** (exponential with :math:`\hat\Lambda = \hat\Omega\hat\Omega^T`):

- **RiemannianDKMSMNG** — :math:`d_\kappa^2 = \exp((\Omega \cdot v)^T \hat\Lambda (\Omega \cdot v)) - 1`
- **RiemannianDKMSLNG** — :math:`d_\kappa^2 = \exp((\Omega_k \cdot v)^T \hat\Lambda (\Omega_k \cdot v)) - 1`
- **RiemannianDKMSTNG** — :math:`d_\kappa^2 = \exp(r^T \hat\Lambda r) - 1`

.. code-block:: python

   from prosemble.core.manifolds import Grassmannian
   from prosemble.models import RiemannianDKRSMNG

   manifold = Grassmannian(4, 2)
   model = RiemannianDKRSMNG(
       manifold=manifold, n_prototypes_per_class=2,
       max_iter=100, lr=0.01, use_scan=False,
   )
   model.fit(X_train, y_train)
   preds = model.predict(X_test)
   print(model.kernel_bandwidths)
   print(model.relevance_profile)

ONNX Export
-----------

All 15 differentiating kernel models support ONNX export.  The three kernel
distance types are implemented as native ONNX subgraphs:

.. code-block:: python

   from prosemble.models import DKGLVQ
   from prosemble.core.onnx_export import export_onnx

   model = DKGLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)
   model.fit(X_train, y_train)

   # Export to ONNX — kernel distance computed in the graph
   onnx_model = export_onnx(model, path='dkglvq.onnx')

Per-prototype bandwidths :math:`\sigma_k` are clamped at export time
(``sigma_min``), and relevance logits are normalized via a Softmax node in
the ONNX graph.  See the :doc:`onnx` guide for full details.

References
----------

.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. *Neurocomputing*, 147,
       83--95.
