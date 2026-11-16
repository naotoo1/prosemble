Neural Gas classifiers
======================

.. currentmodule:: prosemble.models

Supervised Neural Gas models update every prototype of the correct class
for each sample, weighted by its distance rank, instead of only the
nearest one. This neighbourhood cooperation makes training less sensitive
to the initial prototypes, which matters most with many prototypes per
class. The neighbourhood range :math:`\gamma` shrinks during training
from ``gamma_init`` to ``gamma_final``; as :math:`\gamma \to 0` each model
reduces to its LVQ counterpart.

Three costs are available: the GLVQ cost (SNG, SRNG and the matrix
variants), cross-entropy over all classes (the CELVQ-NG family) and the
robust soft LVQ likelihood ratio (the RSLVQ-NG family).

Supervised Neural Gas (SNG)
---------------------------

The simplest Supervised Neural Gas variant. Combines GLVQ loss with Neural Gas
neighborhood cooperation using squared Euclidean distance. All same-class
prototypes are updated per sample, weighted by rank. When :math:`\gamma \to 0`,
SNG recovers standard GLVQ.

.. code-block:: python

   from prosemble.models import SNG

   model = SNG(
       n_prototypes_per_class=3,
       gamma_init=5.0,         # initial neighborhood range
       gamma_final=0.01,       # final (narrower)
       lr_ratio=0.5,           # separate learning rates (Hammer et al. 2003)
       max_iter=100,
       lr=0.01,
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
       gamma_init=5.0,        # initial neighborhood range
       gamma_final=0.01,      # final (narrower)
       lr_ratio=0.5,          # separate learning rates (Hammer et al. 2003)
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

   # Inspect relevances (SRNG learns feature weights like GRLVQ)
   print(model.relevances_)

Supervised Matrix Neural Gas (SMNG, SCMNG, SLNG, STNG)
------------------------------------------------------

The SMNG family extends SRNG with matrix metric adaptation while keeping
Neural Gas neighborhood cooperation. All same-class prototypes participate
in the loss weighted by rank.

All supervised NG models support a ``lr_ratio`` parameter that controls
separate learning rates for correct-class and wrong-class prototypes
(Hammer et al. 2003). Setting ``lr_ratio < 1.0`` reduces the gradient
magnitude flowing through the closest wrong-class prototype:

.. code-block:: python

   from prosemble.models import SMNG

   model = SMNG(
       n_prototypes_per_class=5,
       lr_ratio=0.5,   # wrong-class lr = 0.5 * correct-class lr
       max_iter=200,
       lr=0.01,
   )

When ``lr_ratio=1.0`` (default), correct and wrong-class prototypes
receive equal gradient magnitude.

**SMNG** — Global Omega matrix (like GMLVQ + Neural Gas):

.. code-block:: python

   from prosemble.models import SMNG

   model = SMNG(
       n_prototypes_per_class=3,
       gamma_init=5.0,
       gamma_final=0.01,
       max_iter=200,
       lr=0.01,
   )
   model.fit(X_train, y_train)
   print(model.omega_matrix.shape)   # (n_features, latent_dim)
   print(model.lambda_matrix.shape)  # (n_features, n_features)

**SCMNG** — Per-class Omega matrices (like class-wise GMLVQ + Neural Gas):

.. code-block:: python

   from prosemble.models import SCMNG

   model = SCMNG(
       n_prototypes_per_class=3,
       gamma_init=5.0,
       gamma_final=0.01,
       max_iter=200,
       lr=0.01,
   )
   model.fit(X_train, y_train)
   print(model.omegas_.shape)  # (n_classes, n_features, latent_dim)

Each class shares a single :math:`\Omega_c` matrix, so cooperating same-class
prototypes contribute aligned gradients to the metric. This avoids the gradient
dilution of a global :math:`\Omega` (SMNG) while using fewer parameters than
per-prototype matrices (SLNG).

**SLNG** — Per-prototype Omega matrices (like LGMLVQ + Neural Gas):

.. code-block:: python

   from prosemble.models import SLNG

   model = SLNG(
       n_prototypes_per_class=3,
       gamma_init=5.0,
       gamma_final=0.01,
       max_iter=200,
       lr=0.01,
   )
   model.fit(X_train, y_train)
   print(model.omegas_.shape)  # (n_protos, n_features, latent_dim)

**STNG** — Per-prototype tangent subspaces (like GTLVQ + Neural Gas):

.. code-block:: python

   from prosemble.models import STNG

   model = STNG(
       n_prototypes_per_class=3,
       subspace_dim=2,
       gamma_init=5.0,
       gamma_final=0.01,
       max_iter=200,
       lr=0.01,
   )
   model.fit(X_train, y_train)
   print(model.omegas_.shape)  # (n_protos, n_features, subspace_dim)

Cross-Entropy Neural Gas (CELVQ-NG Family)
------------------------------------------

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

Probabilistic LVQ with Neural Gas (RSLVQ-NG Family)
---------------------------------------------------

The RSLVQ-NG family combines RSLVQ's Gaussian mixture probabilistic
assignment with Neural Gas rank-based neighborhood cooperation. All
prototypes participate in the loss, weighted by both their Gaussian
probability and NG rank.

**RSLVQ_NG** — Euclidean distance (base variant):

.. code-block:: python

   from prosemble.models import RSLVQ_NG

   model = RSLVQ_NG(
       n_prototypes_per_class=3,
       sigma=1.0,
       gamma_init=5.0,
       gamma_final=0.01,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

**MRSLVQ_NG** — Global Omega matrix metric learning:

.. code-block:: python

   from prosemble.models import MRSLVQ_NG

   model = MRSLVQ_NG(
       n_prototypes_per_class=2,
       latent_dim=2,
       sigma=1.0,
       gamma_init=5.0,
       gamma_final=0.01,
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)

   # Learned metric matrices
   print(model.omega_matrix.shape)   # (n_features, latent_dim)
   print(model.lambda_matrix)        # Omega^T @ Omega

**LMRSLVQ_NG** — Per-prototype local Omega matrices:

.. code-block:: python

   from prosemble.models import LMRSLVQ_NG

   model = LMRSLVQ_NG(
       n_prototypes_per_class=2,
       latent_dim=2,
       sigma=1.0,
       gamma_init=5.0,
       gamma_final=0.01,
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)
