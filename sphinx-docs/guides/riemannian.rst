Riemannian Models
=================

Prosemble provides prototype-based learning on Riemannian manifolds.
Prototypes live directly on the manifold and distances are computed via
the intrinsic geodesic metric, preserving the geometric structure of the
data.

Supported Manifolds
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Manifold
     - Points
     - Applications
   * - **SO(n)**
     - :math:`n \times n` rotation matrices
     - Robotics (grasp/pose), 3D vision, structural biology
   * - **SPD(n)**
     - :math:`n \times n` symmetric positive definite matrices
     - EEG/BCI (covariance matrices), diffusion tensor imaging
   * - **Gr(n, k)**
     - :math:`k`-dimensional subspaces of :math:`\mathbb{R}^n`
     - Hyperspectral imaging, video analysis, subspace tracking

All three are available from ``prosemble.core.manifolds``:

.. code-block:: python

   from prosemble.core.manifolds import SO, SPD, Grassmannian

   so3 = SO(3)           # 3x3 rotation matrices
   spd4 = SPD(4)         # 4x4 SPD matrices
   gr5_2 = Grassmannian(5, 2)  # 2D subspaces of R^5

RiemannianSRNG
--------------

Supervised Riemannian Neural Gas.  Combines GLVQ-style margin-based
classification with Neural Gas neighbourhood cooperation on the manifold.

The loss for each sample uses all same-class prototypes, rank-weighted by
:math:`h(k) = \exp(-k / \gamma)`:

.. math::

   \mathcal{L} = \sum_i \sum_j h(k_j) \cdot \sigma\!\left(\frac{d^+(x_i) - d^-(x_i)}{d^+(x_i) + d^-(x_i)}\right)

where :math:`d^+, d^-` are the nearest same-class and different-class
geodesic distances.

.. code-block:: python

   from prosemble.models import RiemannianSRNG
   from prosemble.core.manifolds import SO
   import jax.numpy as jnp

   # Generate synthetic rotation data
   key = jax.random.PRNGKey(0)
   manifold = SO(3)
   X = jnp.stack([manifold.random_point(jax.random.fold_in(key, i))
                   for i in range(40)])
   X = X.reshape(40, -1)  # flatten to (n_samples, 9)
   y = jnp.array([0] * 20 + [1] * 20)

   model = RiemannianSRNG(
       manifold=manifold,
       n_prototypes_per_class=2,
       max_iter=50,
       lr=0.01,
       gamma_final=0.01,
   )
   model.fit(X, y)
   labels = model.predict(X)

RiemannianSMNG
--------------

Supervised Matrix Neural Gas on manifolds.  Adds a global metric
adaptation matrix :math:`\Omega` that operates in the tangent space at
each prototype:

.. math::

   d(x, w_k) = \|\Omega \cdot \text{Log}_{w_k}(x)\|^2

where :math:`\text{Log}_{w_k}` is the Riemannian logarithmic map
(tangent vector from :math:`w_k` to :math:`x`).

.. code-block:: python

   from prosemble.models import RiemannianSMNG
   from prosemble.core.manifolds import SPD

   manifold = SPD(3)
   model = RiemannianSMNG(
       manifold=manifold,
       n_prototypes_per_class=2,
       latent_dim=4,
       max_iter=50,
       lr=0.01,
   )
   model.fit(X, y)

   # Learned relevance matrix
   print(model.relevance_matrix().shape)

RiemannianSLNG
--------------

Supervised Localized Matrix Neural Gas.  Each prototype :math:`w_k` has
its own metric matrix :math:`\Omega_k`:

.. math::

   d(x, w_k) = \|\Omega_k \cdot \text{Log}_{w_k}(x)\|^2

This allows different prototypes to focus on different tangent directions,
useful when the discriminative structure varies across the manifold.

.. code-block:: python

   from prosemble.models import RiemannianSLNG
   from prosemble.core.manifolds import Grassmannian

   manifold = Grassmannian(5, 2)
   model = RiemannianSLNG(
       manifold=manifold,
       n_prototypes_per_class=2,
       latent_dim=3,
       max_iter=50,
       lr=0.01,
   )
   model.fit(X, y)

RiemannianSTNG
--------------

Supervised Tangent Neural Gas.  Each prototype has a tangent subspace
:math:`\Omega_k`, and distance is measured in the complement of that
subspace (the residual after projection):

.. math::

   d(x, w_k) = \|\text{Log}_{w_k}(x) - \Omega_k \Omega_k^T \text{Log}_{w_k}(x)\|^2

This captures invariance structure — directions spanned by
:math:`\Omega_k` are ignored.

.. code-block:: python

   from prosemble.models import RiemannianSTNG
   from prosemble.core.manifolds import SO

   manifold = SO(3)
   model = RiemannianSTNG(
       manifold=manifold,
       n_prototypes_per_class=2,
       subspace_dim=2,
       max_iter=50,
       lr=0.01,
   )
   model.fit(X, y)

RiemannianNeuralGas
-------------------

Unsupervised Neural Gas on Riemannian manifolds.  Distributes prototypes
to match the data density using rank-based cooperation with geodesic
distances and exponential map updates.

.. code-block:: python

   from prosemble.models import RiemannianNeuralGas
   from prosemble.core.manifolds import SO

   manifold = SO(3)
   model = RiemannianNeuralGas(
       manifold=manifold,
       n_prototypes=5,
       max_iter=50,
       lr_init=0.3,
       lr_final=0.01,
       lambda_final=0.01,
   )
   model.fit(X)

   labels = model.predict(X)       # nearest prototype assignment
   distances = model.transform(X)  # geodesic distance matrix

.. note::

   All Riemannian models use **Python loops** (not ``lax.scan``) because
   manifold projection after each gradient step is not compatible with
   JAX's functional loop primitives.

Choosing a Model
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Model
     - Metric
     - Best For
   * - RiemannianSRNG
     - Geodesic distance
     - General manifold classification
   * - RiemannianSMNG
     - Global :math:`\Omega` in tangent space
     - Feature selection on manifolds
   * - RiemannianSLNG
     - Per-prototype :math:`\Omega_k`
     - Heterogeneous tangent structure
   * - RiemannianSTNG
     - Tangent subspace projection
     - Invariance learning on manifolds
   * - RiemannianNeuralGas
     - Geodesic distance
     - Unsupervised manifold clustering
