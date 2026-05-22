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

References
----------

.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. *Neurocomputing*, 147,
       83--95.
