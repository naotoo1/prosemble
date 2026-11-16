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

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Manifold
     - Points
     - Applications
   * - **HyperbolicPoincare(d)**
     - Vectors in :math:`\mathbb{B}^d = \{x \in \mathbb{R}^d : \|x\| < 1\}`
     - Hierarchical data, taxonomy embeddings, NLP trees

All four are available from ``prosemble.core.manifolds``:

.. code-block:: python

   from prosemble.core.manifolds import SO, SPD, Grassmannian, HyperbolicPoincare

   so3 = SO(3)           # 3x3 rotation matrices
   spd4 = SPD(4)         # 4x4 SPD matrices
   gr5_2 = Grassmannian(5, 2)  # 2D subspaces of R^5
   hyp8 = HyperbolicPoincare(8)  # 8D Poincare ball

HyperbolicPoincare
~~~~~~~~~~~~~~~~~~

The Poincare ball model of hyperbolic geometry.  Points live in the open
unit ball :math:`\mathbb{B}^d = \{x \in \mathbb{R}^d : \|x\| < 1\}` with
the Riemannian metric tensor :math:`g_x = \lambda_x^2 I` where
:math:`\lambda_x = 2/(1 - \|x\|^2)` is the conformal factor.

The geodesic distance is:

.. math::

   d(x, y) = \text{arcosh}\!\left(1 + \frac{2\|x - y\|^2}{(1 - \|x\|^2)(1 - \|y\|^2)}\right)

Prototype updates use the exponential and logarithmic maps based on
Mobius addition:

.. math::

   \text{Exp}_x(v) = x \oplus \left(\tanh\!\left(\frac{\lambda_x \|v\|}{2}\right) \cdot \frac{v}{\|v\|}\right)

.. math::

   \text{Log}_x(y) = \frac{2}{\lambda_x} \text{arctanh}(\|-x \oplus y\|) \cdot \frac{-x \oplus y}{\|-x \oplus y\|}

Hyperbolic space is particularly suited for data with hierarchical or
tree-like structure, as distances grow exponentially toward the boundary.

.. code-block:: python

   from prosemble.models import RiemannianSRNG
   from prosemble.core.manifolds import HyperbolicPoincare
   import jax
   import jax.numpy as jnp

   manifold = HyperbolicPoincare(4)

   # Generate data inside the Poincare ball
   key = jax.random.PRNGKey(0)
   X = jax.random.normal(key, (40, 4)) * 0.3
   X = jax.vmap(manifold.project)(X)  # ensure points lie in ball
   X = X.reshape(40, -1)
   y = jnp.array([0] * 20 + [1] * 20)

   model = RiemannianSRNG(
       manifold=manifold,
       n_prototypes_per_class=2,
       max_iter=50,
       lr=0.01,
   )
   model.fit(X, y)
   labels = model.predict(X)

All supervised Riemannian models (RiemannianSRNG, RiemannianSMNG,
RiemannianSLNG, RiemannianSTNG) and the unsupervised RiemannianNeuralGas
work with ``HyperbolicPoincare`` without any code changes.

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

RiemannianRSLVQ
---------------

Riemannian Robust Soft LVQ.  A labelled mixture on the manifold with
components :math:`p(x | k) \propto \exp(-d^2(x, w_k) / 2\sigma^2)`,
trained by maximising the likelihood ratio

.. math::

   \sum_i \log
   \frac{\sum_{k: c(k) = y_i} \exp(-d^2(x_i, w_k) / 2\sigma^2)}
        {\sum_k \exp(-d^2(x_i, w_k) / 2\sigma^2)}

On SPD matrices with the affine-invariant metric this is the model of
Tang et al. (2021).  Here it is available on all four manifolds, the
width :math:`\sigma` can be learned, and prototypes move along geodesics
as described under *Prototype Updates* below.  On SO(n) the distance is
the chordal one, which makes each component a matrix Fisher
distribution centred at its prototype.

.. code-block:: python

   from prosemble.models import RiemannianRSLVQ
   from prosemble.core.manifolds import SPD

   model = RiemannianRSLVQ(
       manifold=SPD(4),
       n_prototypes_per_class=2,
       sigma='auto',          # initial width from the data
       adapt_sigma=True,      # learn it during training
       max_iter=100,
       lr=0.01,
   )
   model.fit(X, y)

   proba = model.predict_proba(X)          # class posteriors
   labels = model.predict_with_rejection(X, confidence=0.8)

RiemannianMRSLVQ
----------------

RiemannianRSLVQ with a learned metric in the tangent space.  A sample
is seen from prototype :math:`w_k` through its tangent vector
:math:`\operatorname{Log}_{w_k}(x)`, written in orthonormal coordinates
:math:`u_k(x)` of the tangent space, and each component is a Gaussian in
these coordinates:

.. math::

   p(x | k) \propto \exp\left(
       -\frac{u_k(x)^T \Lambda\, u_k(x)}{2\sigma^2}\right),
   \qquad \Lambda = \Omega \Omega^T

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Manifold
     - Coordinates :math:`u_k(x)`
   * - SO(n)
     - :math:`w_k^T \operatorname{Log}_{w_k}(x) / \sqrt2`
   * - SPD(n)
     - :math:`\log(w_k^{-1/2}\, x\, w_k^{-1/2})`
   * - Grassmannian
     - :math:`\operatorname{Log}_{w_k}(x)`
   * - Poincaré ball
     - :math:`\lambda_{w_k} \operatorname{Log}_{w_k}(x)`

With the exact log map :math:`\|u_k(x)\|` is the geodesic distance, so
:math:`\Lambda = I`, where training starts, is the isotropic component of
RiemannianRSLVQ.  One matrix is learned for all prototypes, more slowly
than the prototypes (``matrix_lr_ratio``).

.. code-block:: python

   from prosemble.models import RiemannianMRSLVQ
   from prosemble.core.manifolds import SPD

   model = RiemannianMRSLVQ(
       manifold=SPD(4),
       n_prototypes_per_class=2,
       latent_dim=None,       # columns of Omega; default: 16 for SPD(4)
       matrix_lr_ratio=0.3,   # the matrix learns at 0.3 * lr
       max_iter=100,
       lr=0.01,
   )
   model.fit(X, y)
   print(model.lambda_matrix.shape)        # (16, 16)

The matrix acts on the flattened coordinates and adds parameters.  When
the classes are isotropic around their centres it has nothing to learn,
and RiemannianRSLVQ is the better choice.  On synthetic data whose
classes differ along a few tangent directions the matrix models were
more accurate than RiemannianRSLVQ in the Poincaré ball and on the
Grassmannian, and not on SPD(n) or SO(n), so compare both on your data.

Prototype Updates
-----------------

All supervised Riemannian models move their prototypes intrinsically
(``update_rule='riemannian'``, the default).  One training step is:

1. The Euclidean gradient :math:`G` of the loss is converted to the
   Riemannian gradient, which accounts for the metric of the manifold,
   and written in an orthonormal frame of the tangent space at the
   prototype.  In these coordinates the metric is the ordinary Euclidean
   one, so a step of length :math:`\eta` is a geodesic step of length
   :math:`\eta` anywhere on the manifold.
2. The optimizer (``adam`` by default) computes its step :math:`v` in
   these coordinates.
3. The step is limited to :math:`\|v\|_w \le \tau \cdot \text{inj}(w)`,
   a fraction ``tau`` of the injectivity radius, inside which the
   exponential map is one-to-one.
4. The prototype follows the geodesic:
   :math:`w \leftarrow \text{Exp}_w(v)`.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Manifold
     - Riemannian gradient
     - Orthonormal coordinates
   * - **SO(n)**
     - :math:`2 R\,\text{skew}(R^T G)`
     - :math:`\sqrt{2}\,\text{skew}(R^T G)`
   * - **SPD(n)**
     - :math:`W\,\text{sym}(G)\,W`
     - :math:`W^{1/2}\,\text{sym}(G)\,W^{1/2}`
   * - **Gr(n, k)**
     - :math:`(I - Q Q^T) G`
     - :math:`(I - Q Q^T) G`
   * - **HyperbolicPoincare(d)**
     - :math:`G / \lambda_x^2`
     - :math:`G / \lambda_x`

With ``optimizer='sgd'`` this is Riemannian gradient descent,
:math:`w \leftarrow \text{Exp}_w(-\eta\,\text{grad} f(w))`.

A consequence is that training respects the symmetries of the metric.
On SPD matrices, for example, multiplying all data by a constant
multiplies the learned prototypes by the same constant and changes
nothing else.

``update_rule='projected'`` selects the simpler alternative: an
ordinary Euclidean optimizer step followed by ``manifold.project``.
Models saved before ``update_rule`` existed load with this setting.

.. code-block:: python

   model = RiemannianSRNG(
       manifold=SPD(4),
       n_prototypes_per_class=2,
       update_rule='riemannian',   # default
       tau=0.95,                   # step limit, fraction of injectivity radius
   )

A custom manifold supports the intrinsic update if, in addition to the
:class:`~prosemble.core.protocols.Manifold` protocol, it provides
``gradient_to_frame(point, euclidean_grad)``,
``frame_to_tangent(point, coords)`` and ``tangent_norm(point, tangent)``.
Otherwise the model warns and uses ``'projected'``.

Tangent Vectors
---------------

The matrix and tangent variants (RiemannianSMNG, RiemannianSLNG,
RiemannianSTNG and the differentiating-kernel models built on them)
measure distance on the tangent vector :math:`\text{Log}_{w_k}(x)` from a
prototype to a sample.  The ``log_map`` option selects how it is computed
on SO(n) and on the Grassmannian.

``log_map='exact'`` is the Riemannian log map, and the default on SO(n).
Its length is the geodesic distance, and on the Grassmannian it depends
only on the subspace, not on the basis used to represent it:

.. math::

   \text{Log}_R(S) = R \log(R^T S), \qquad
   \text{Log}_{Q_1}(Q_2) = B A^T \, \Phi(A A^T),
   \quad A = Q_1^T Q_2, \; B = (I - Q_1 Q_1^T) Q_2,

with :math:`\Phi(\cos^2\theta) = \theta / (\sin\theta \cos\theta)` applied
to the principal angles.

``log_map='first_order'`` is the approximation for nearby points,
:math:`R\,\text{skew}(R^T S)` and :math:`Q_2 - Q_1 Q_1^T Q_2`, and the
default on the Grassmannian.  Its length is :math:`\sin\theta` rather
than :math:`\theta`:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Rotation angle on SO(3)
     - Exact length
     - First-order length
   * - 30°
     - 0.52
     - 0.50
   * - 90°
     - 1.57
     - 1.00
   * - 150°
     - 2.62
     - 0.50
   * - 175°
     - 3.05
     - 0.09

(lengths in units of the geodesic distance).  With the first-order form
a rotation by 150° is as close as one by 30°, and on the Grassmannian a
different basis for the same subspace gives a different vector.

The log map is discontinuous at the cut locus: a rotation by 180°, or a
principal angle of 90°.  Within about 0.1° of it the exact vector is
scaled down continuously to zero.

.. code-block:: python

   model = RiemannianSMNG(
       manifold=SO(3),
       n_prototypes_per_class=2,
       log_map='exact',          # default on SO(n)
   )

   model = RiemannianSMNG(
       manifold=Grassmannian(6, 2),
       n_prototypes_per_class=2,
       log_map='exact',          # opt-in on the Grassmannian
   )

On the Grassmannian the first-order form stays the default for two
reasons.  The sine of a principal angle is still increasing up to 90°, so
distant subspaces do not appear close.  And the exact log map needs an
eigendecomposition, which ONNX does not provide.  It is worth selecting
when samples of one subspace may come with different bases.

On SO(2) and SO(3) the exact log map has a closed form and is exported to
ONNX.  To export a model on SO(n) with n > 3, train it with
``log_map='first_order'``.  Models saved before ``log_map`` existed load
with ``'first_order'``.  The log maps on SPD and hyperbolic manifolds are
exact in either case.

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

.. list-table:: Supported Manifold Selection
   :header-rows: 1
   :widths: 25 35 40

   * - Manifold
     - Geometry
     - Use When
   * - SO(n)
     - Rotation matrices
     - Pose estimation, robotics, structural biology
   * - SPD(n)
     - Positive definite matrices
     - Covariance matrices (EEG/BCI, DTI)
   * - Grassmannian(n, k)
     - Subspaces of :math:`\mathbb{R}^n`
     - Video/image subspace analysis
   * - HyperbolicPoincare(d)
     - Poincare ball :math:`\mathbb{B}^d`
     - Hierarchical/tree-structured data
