Fuzzy Clustering
================

Prosemble provides 15 fuzzy clustering algorithms, all implemented in JAX
with JIT compilation and ``lax.scan`` training loops. Every model follows
the same ``fit`` / ``predict`` / ``predict_proba`` API.

FCM — Fuzzy C-Means
--------------------

The baseline fuzzy clustering algorithm. Each sample has a membership
degree to every cluster, controlled by the fuzzifier :math:`m`.

.. code-block:: python

   from prosemble.models import FCM
   from prosemble.datasets import load_iris_jax

   dataset = load_iris_jax()
   X = dataset.input_data

   model = FCM(
       n_clusters=3,
       fuzzifier=2.0,       # m > 1; higher = fuzzier
       max_iter=100,
       epsilon=1e-5,        # convergence tolerance
       random_seed=42,
   )
   model.fit(X)

   # Hard labels
   labels = model.predict(X)

   # Fuzzy membership matrix U (n_samples x n_clusters)
   U = model.predict_proba(X)

   # Centroids
   print(model.centroids_.shape)      # (n_clusters, n_features)
   print(f"Iterations: {model.n_iter_}")
   print(f"Objective: {model.objective_:.4f}")

PCM — Possibilistic C-Means
----------------------------

Relaxes the FCM constraint that membership rows sum to 1. Each membership
value represents a "typicality" — how typical a sample is for a cluster,
independent of other clusters.

.. code-block:: python

   from prosemble.models import PCM

   model = PCM(
       n_clusters=3,
       fuzzifier=2.0,
       max_iter=100,
       init_method='fcm',   # initialize from FCM solution
   )
   model.fit(X)

FPCM — Fuzzy Possibilistic C-Means
-----------------------------------

Maintains **two** membership matrices simultaneously:

- **U** (fuzzy): rows sum to 1, standard FCM constraint
- **T** (typicality): columns sum to 1 (Pal, Pal & Bezdek, 1997)

.. code-block:: python

   from prosemble.models import FPCM

   model = FPCM(
       n_clusters=3,
       fuzzifier=2.0,       # controls U fuzziness
       eta=2.0,             # controls T fuzziness
       max_iter=100,
   )
   model.fit(X)

   print(model.U_.shape)   # fuzzy membership
   print(model.T_.shape)   # typicality

PFCM — Possibilistic Fuzzy C-Means
-----------------------------------

Combines FCM and PCM with explicit weighting parameters ``a`` and ``b``.

.. code-block:: python

   from prosemble.models import PFCM

   model = PFCM(
       n_clusters=3,
       fuzzifier=2.0,
       eta=2.0,
       a=1.0,               # weight for FCM term
       b=1.0,               # weight for PCM term
       max_iter=100,
   )
   model.fit(X)

AFCM — Adaptive Fuzzy C-Means
------------------------------

Extends FCM with per-cluster adaptive weights that balance local vs global
influence.

.. code-block:: python

   from prosemble.models import AFCM

   model = AFCM(n_clusters=3, fuzzifier=2.0, max_iter=100)
   model.fit(X)

HCM — Hard C-Means
-------------------

Crisp (non-fuzzy) clustering. Equivalent to K-Means but implemented in
the prosemble framework with ``lax.scan`` training.

.. code-block:: python

   from prosemble.models import HCM

   model = HCM(n_clusters=3, max_iter=100)
   model.fit(X)
   labels = model.predict(X)  # hard labels only

KMeans++ — Smart Initialization
--------------------------------

K-Means with the K-Means++ initialization algorithm for better convergence.

.. code-block:: python

   from prosemble.models import KMeansPlusPlus

   model = KMeansPlusPlus(n_clusters=3, max_iter=100, random_seed=42)
   model.fit(X)

IPCM / IPCM2 — Improved PCM
-----------------------------

Two-phase algorithms that improve PCM by alternating between FCM and PCM
phases to avoid the coincident cluster problem.

.. code-block:: python

   from prosemble.models import IPCM, IPCM2

   model = IPCM(n_clusters=3, fuzzifier=2.0, max_iter=100)
   model.fit(X)

Kernel Variants
---------------

Kernel variants (prefix ``K``) operate in kernel-induced feature space
using Gaussian kernels, enabling nonlinear cluster boundaries.

.. code-block:: python

   from prosemble.models import KFCM, KPCM, KFPCM, KPFCM, KAFCM, KIPCM, KIPCM2

   model = KFCM(
       n_clusters=3,
       fuzzifier=2.0,
       sigma=1.0,           # Gaussian kernel bandwidth
       max_iter=100,
   )
   model.fit(X)

All kernel variants accept a ``sigma`` parameter controlling the kernel
bandwidth. Smaller sigma = sharper kernel (more local), larger sigma = smoother.

.. list-table:: Kernel Variants
   :header-rows: 1
   :widths: 20 40 40

   * - Model
     - Base Algorithm
     - Key Difference
   * - ``KFCM``
     - FCM
     - Kernel fuzzy membership
   * - ``KPCM``
     - PCM
     - Kernel possibilistic membership
   * - ``KFPCM``
     - FPCM
     - Kernel fuzzy + typicality
   * - ``KPFCM``
     - PFCM
     - Kernel possibilistic-fuzzy
   * - ``KAFCM``
     - AFCM
     - Kernel adaptive FCM
   * - ``KIPCM``
     - IPCM
     - Kernel improved PCM
   * - ``KIPCM2``
     - IPCM2
     - Kernel improved PCM (variant 2)

Common Patterns
---------------

**Resume training:**

.. code-block:: python

   model.fit(X, max_iter=50)
   model.fit(X, resume=True)  # continue from last state

**Custom initial centroids:**

.. code-block:: python

   import jax.numpy as jnp
   initial = jnp.array([[5.0, 3.5, 1.4, 0.2],
                         [6.3, 2.8, 5.1, 1.5],
                         [5.8, 2.7, 4.2, 1.3]])
   model.fit(X, initial_centroids=initial)

**Objective history:**

.. code-block:: python

   import matplotlib.pyplot as plt
   plt.plot(model.objective_history_)
   plt.xlabel('Iteration')
   plt.ylabel('Objective')
   plt.title('Convergence')

**Fitted attributes** (after ``fit``):

- ``model.centroids_`` — cluster centers
- ``model.U_`` — fuzzy membership matrix (FCM-family)
- ``model.T_`` — typicality matrix (FPCM, PFCM)
- ``model.n_iter_`` — iterations run
- ``model.objective_`` — final objective value
- ``model.objective_history_`` — objective per iteration
