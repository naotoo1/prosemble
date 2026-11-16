Classifying orientations
========================

.. currentmodule:: prosemble.models

Robot grasps, camera poses and molecular conformations are described by
orientations: rotations of three-dimensional space. Rotations form the
curved space SO(3). This tutorial classifies orientations with
:class:`RiemannianSRNG` on SO(3) and compares two common ways of turning
rotations into vectors.

The data
--------

Each class is a cloud of orientations around a rotation by 170° about
the x, y or z axis, scattered by random rotations of 46° on average:

.. code-block:: python

   import numpy as np

   def rotation(w):
       """Rotation matrix for the axis-angle vector w (Rodrigues' formula)."""
       theta = np.linalg.norm(w)
       K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
       if theta < 1e-12:
           return np.eye(3) + K
       return np.eye(3) + np.sin(theta) / theta * K + (1 - np.cos(theta)) / theta ** 2 * K @ K

   def orientation_samples(n_per_class, seed, spread=0.5, angle=170):
       means = [rotation(np.deg2rad(angle) * axis) for axis in np.eye(3)]
       rng = np.random.default_rng(seed)
       R, y = [], []
       for c, M in enumerate(means):
           for _ in range(n_per_class):
               R.append(M @ rotation(spread * rng.normal(size=3)))
               y.append(c)
       return np.array(R, dtype=np.float32), np.array(y, dtype=np.int32)

   R_train, y_train = orientation_samples(40, seed=40000)
   R_test, y_test = orientation_samples(200, seed=99)

Train on SO(3)
--------------

The Riemannian models take each 3 × 3 rotation matrix flattened to a row
of nine values:

.. code-block:: python

   import jax.numpy as jnp
   from prosemble.core.manifolds import SO
   from prosemble.models import RiemannianSRNG

   model = RiemannianSRNG(manifold=SO(3), n_prototypes_per_class=1,
                          max_iter=300, lr=0.05, random_seed=0)
   model.fit(jnp.asarray(R_train.reshape(-1, 9)), jnp.asarray(y_train))

   pred = np.asarray(model.predict(jnp.asarray(R_test.reshape(-1, 9))))
   np.mean(pred == y_test)                  # 1.0 (0.997 on average over five training sets)

The distance is the rotation angle between two orientations, and the
prototypes are moved along geodesics of SO(3).

Rotations as vectors
--------------------

.. list-table:: Test accuracy, mean over five training sets
   :header-rows: 1
   :widths: 28 24 24 24

   * - Samples per class
     - GLVQ on quaternions
     - GLVQ on rotation matrices
     - RiemannianSRNG on SO(3)
   * - 10
     - 0.526
     - 0.985
     - 0.996
   * - 40
     - 0.558
     - 0.993
     - 0.997

.. figure:: /_static/figures/tutorial-rotations-light.png
   :class: only-light pr-figure

.. figure:: /_static/figures/tutorial-rotations-dark.png
   :class: only-dark pr-figure

   Left: each test orientation applied to the direction (1, 1, 1)/√3, with
   the three learned prototypes drawn as coordinate frames. Right: the
   quaternions of the same orientations.

**Quaternions.** A unit quaternion :math:`q` and its negative :math:`-q`
describe the same rotation, and a common convention keeps the scalar part
:math:`w \ge 0`. For rotations near 180°, :math:`w` is close to zero, and
the convention sends half of each class to :math:`q` and the other half to
:math:`-q` (right). Two orientations a few degrees apart then have
quaternions far apart, and a Euclidean classifier fails.

**Rotation matrices.** The Euclidean distance between two rotation
matrices grows with the angle between them, so GLVQ on the nine entries
classifies almost as well as the Riemannian model. Its prototypes,
however, are averages of matrices and are not rotations: for them
:math:`\|W^\top W - I\|` is about 1.0, against :math:`10^{-7}` for the
prototypes of :class:`RiemannianSRNG`. A Riemannian prototype is an
orientation that can be applied to an object, shown or compared with
others as such.

:doc:`/user_guide/riemannian` covers SO(n) in general and the other
manifolds.
