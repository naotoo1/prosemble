Classifying subspaces
=====================

.. currentmodule:: prosemble.models

Sometimes a sample is a set of vectors rather than one: several images of
a face, the frames of a video, the spectra of a region. A common
description of such a set is the subspace its vectors span. Subspaces of
dimension :math:`k` in :math:`\mathbb{R}^n` form the Grassmannian
Gr(n, k), and this tutorial classifies them with :class:`RiemannianSRNG`.

The data
--------

Each class has its own two-dimensional subspace of :math:`\mathbb{R}^{10}`.
A sample is a set of eight noisy vectors from its class subspace, described
by the basis that a singular value decomposition returns:

.. code-block:: python

   import numpy as np

   def subspace_sets(n_per_class, seed, dim=10, k=2, size=8, noise=0.8, spread=0.6):
       rng0 = np.random.default_rng(0)
       common = np.linalg.qr(rng0.normal(size=(dim, k)))[0]
       classes = [np.linalg.qr(common + spread * rng0.normal(size=(dim, k)))[0]
                  for _ in range(3)]
       rng = np.random.default_rng(seed)
       bases, labels = [], []
       for c, U in enumerate(classes):
           for _ in range(n_per_class):
               V = U @ rng.normal(size=(k, size)) + noise * rng.normal(size=(dim, size))
               bases.append(np.linalg.svd(V, full_matrices=False)[0][:, :k])
               labels.append(c)
       return np.array(bases, dtype=np.float32), np.array(labels, dtype=np.int32)

   B_train, y_train = subspace_sets(40, seed=40000)
   B_test, y_test = subspace_sets(200, seed=99)

A basis is not unique: rotating or flipping its vectors within the plane
gives another basis of the same subspace, and the SVD returns whichever
one the data happen to produce.

Train on the Grassmannian
-------------------------

.. code-block:: python

   import jax.numpy as jnp
   from prosemble.core.manifolds import Grassmannian
   from prosemble.models import RiemannianSRNG

   model = RiemannianSRNG(manifold=Grassmannian(10, 2), n_prototypes_per_class=1,
                          max_iter=300, lr=0.05, random_seed=0)
   model.fit(jnp.asarray(B_train.reshape(len(B_train), -1)), jnp.asarray(y_train))

The distance between two subspaces is computed from their principal angles
and does not depend on the bases. The prototypes are subspaces too, each
given by an orthonormal basis.

Bases, projections and subspaces
--------------------------------

.. figure:: /_static/figures/tutorial-subspaces-light.png
   :class: only-light pr-figure

.. figure:: /_static/figures/tutorial-subspaces-dark.png
   :class: only-dark pr-figure

   Left: one plane through the origin with two different bases. Right:
   test accuracy (mean and standard deviation over five training sets)
   against the number of training sets per class.

GLVQ on the flattened bases sees two bases of the same plane as different
points, and stays below 0.5 accuracy, not far above chance (1/3). Two
representations remove the ambiguity:

* the projection matrix :math:`UU^\top`, which is the same for every
  basis :math:`U` of the subspace, used as a vector of 100 values;
* the Grassmannian itself, with :class:`RiemannianSRNG`.

Both reach 0.85 to 0.89 with 40 to 80 sets per class. On this data GLVQ on
the projection matrices is slightly more accurate (0.888 against 0.854
with 80 sets per class); the Riemannian model works with far fewer numbers
(a 10 × 2 basis instead of a 10 × 10 matrix) and its prototypes are
subspaces that can be compared through their principal angles.

.. list-table:: Test accuracy, mean over five training sets
   :header-rows: 1
   :widths: 25 25 25 25

   * - Sets per class
     - GLVQ on the bases
     - GLVQ on the projections
     - RiemannianSRNG on Gr(10, 2)
   * - 10
     - 0.384
     - 0.799
     - 0.777
   * - 40
     - 0.459
     - 0.869
     - 0.858
   * - 80
     - 0.475
     - 0.888
     - 0.854

:class:`RiemannianRSLVQ`, with a single width for all its Gaussian
components, reached 0.77 with 40 sets per class here; the GLVQ-type cost
of :class:`RiemannianSRNG` suited these classes better.
