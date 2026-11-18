Classifying covariance matrices
===============================

.. currentmodule:: prosemble.models

In EEG and brain–computer interfaces, radar, and image analysis, a sample
is often a covariance matrix: how the channels of a multichannel recording
vary together. Covariance matrices are symmetric positive definite (SPD),
and the set of SPD matrices is curved: the straight line between two of
them, and the Euclidean distance along it, ignore that structure. This
tutorial compares a classifier that treats the matrices as plain vectors
with :class:`RiemannianRSLVQ`, which measures distances along the curved
space SPD(n).

The data
--------

The recordings are simulated. Six hidden sources are mixed into six
channels by a fixed matrix. In class :math:`k`, source :math:`k` has about
3.3 times more power than the others. Every recording also has a random
overall gain, as real recordings do, and each sample is the 6 × 6
covariance matrix of one recording:

.. code-block:: python

   import numpy as np

   def covariance_trials(n_per_class, seed, channels=6, length=200):
       mixing = np.random.default_rng(0).normal(size=(channels, channels))
       log_power = np.zeros((3, channels))
       log_power[[0, 1, 2], [0, 1, 2]] = 1.2
       rng = np.random.default_rng(seed)
       covs, labels = [], []
       for k in range(3):
           for _ in range(n_per_class):
               power = np.exp(log_power[k] + rng.normal(0, 0.4, channels)
                              + rng.normal(0, 0.8))         # overall gain
               sources = rng.normal(size=(channels, length)) * np.sqrt(power)[:, None]
               x = mixing @ sources + 0.1 * rng.normal(size=(channels, length))
               covs.append(x @ x.T / length)
               labels.append(k)
       return np.array(covs, dtype=np.float32), np.array(labels, dtype=np.int32)

   P_train, y_train = covariance_trials(40, seed=40000)   # 40 per class
   P_test, y_test = covariance_trials(200, seed=99)

Train on the manifold
---------------------

The Riemannian models take each matrix flattened to a row of
:math:`n \times n` values, and the manifold that tells them how to read it:

.. code-block:: python

   import jax.numpy as jnp
   from prosemble.core.manifolds import SPD
   from prosemble.models import RiemannianRSLVQ

   X_train = jnp.asarray(P_train.reshape(len(P_train), -1))
   X_test = jnp.asarray(P_test.reshape(len(P_test), -1))

   model = RiemannianRSLVQ(manifold=SPD(6), n_prototypes_per_class=1,
                           sigma='auto', adapt_sigma=True,
                           max_iter=300, lr=0.01, random_seed=0)
   model.fit(X_train, jnp.asarray(y_train))

   np.mean(np.asarray(model.predict(X_test)) == y_test)     # 0.958

The distance is the affine-invariant one,
:math:`d(A, B) = \|\log(A^{-1/2} B A^{-1/2})\|_F`. The prototypes are
covariance matrices themselves, ``model.prototypes_.reshape(-1, 6, 6)``,
and the model gives class probabilities that can be calibrated and used to
reject uncertain samples (:doc:`/user_guide/probabilities`).

For comparison, :class:`GLVQ` on the same matrices flattened into vectors
of 36 standardised values classifies 88.8 % of the test recordings
correctly, against 95.8 % for the Riemannian model.

Why the geometry matters
------------------------

.. figure:: /_static/figures/tutorial-spd-distances-light.png
   :class: only-light pr-figure

.. figure:: /_static/figures/tutorial-spd-distances-dark.png
   :class: only-dark pr-figure

   Each point is a test recording: its squared distance to the prototype
   of its own class against its distance to the nearest prototype of
   another class. Below the diagonal it is classified correctly.

With the Euclidean distance (left), recordings with a high overall gain
lie far from every prototype and on the diagonal: the gain swamps the
difference between the classes. The Riemannian distance compares matrices
on a logarithmic scale: a gain changes a recording's distances to all the
prototypes by similar amounts instead of multiplying them, and most
recordings fall below the diagonal (right).

A strong alternative
--------------------

Taking the matrix logarithm of every covariance matrix maps SPD(n) to a
flat space in which Euclidean distances are meaningful (the log-Euclidean
metric). :class:`GLVQ` on these logarithms is a cheap and strong baseline,
and it is worth trying next to the Riemannian model:

.. code-block:: python

   def logm(P):
       w, V = np.linalg.eigh(P)
       return (V * np.log(w)[..., None, :]) @ np.swapaxes(V, -1, -2)

.. figure:: /_static/figures/tutorial-spd-curve-light.png
   :class: only-light pr-figure
   :width: 75%
   :align: center

.. figure:: /_static/figures/tutorial-spd-curve-dark.png
   :class: only-dark pr-figure
   :width: 75%
   :align: center

   Test accuracy (mean and standard deviation over five training sets) as
   the number of training recordings grows.

.. list-table:: Test accuracy, mean over five training sets
   :header-rows: 1
   :widths: 25 25 25 25

   * - Recordings per class
     - GLVQ, raw matrices
     - GLVQ, matrix logarithms
     - RiemannianRSLVQ
   * - 5
     - 0.840
     - 0.927
     - 0.911
   * - 10
     - 0.794
     - 0.925
     - 0.897
   * - 20
     - 0.869
     - 0.952
     - 0.935
   * - 40
     - 0.889
     - 0.960
     - 0.952
   * - 80
     - 0.880
     - 0.959
     - 0.961

Both geometry-aware approaches are 7 to 10 points more accurate than
treating the matrices as vectors. On this data the log-Euclidean baseline
is as accurate as the Riemannian model, and steadier with very few
recordings per class. The Riemannian model works on the matrices
themselves: its prototypes are covariance matrices, and its distance is
unchanged when all matrices are transformed by the same invertible map
(for example a different mixing of the channels), which the log-Euclidean
distance is not.

Settings
--------

* Prototype updates move along geodesics of the manifold. Keep the
  learning rate small (``lr=0.01`` here). On training data that it
  separates, the model keeps moving its prototypes away from the data to
  widen its margins; after a long run with a large ``lr`` they can become
  too ill-conditioned for float32, and training stops with a warning and
  keeps the parameters with the lowest training loss.
* ``prototype_prior=True`` keeps each prototype near the Karcher mean of
  its class, so the prototypes stay typical covariance matrices of their
  classes and training cannot diverge this way. On this data it was less
  accurate with 5 recordings per class (0.863 against 0.903 without it,
  mean over 30 training sets) and slightly more accurate with 40; see
  :doc:`/user_guide/riemannian`.
* With ``adapt_sigma=True`` the model also learns the width of its
  Gaussian components; ``sigma='auto'`` sets the initial width from the
  data.

:doc:`/user_guide/riemannian` describes the other manifolds (rotations,
subspaces, hyperbolic space) and the other Riemannian models.
