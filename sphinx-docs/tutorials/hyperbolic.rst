Hierarchies in hyperbolic space
===============================

.. currentmodule:: prosemble.models

Trees, taxonomies and other hierarchies grow exponentially with depth, and
so does hyperbolic space: a tree can be embedded in a hyperbolic plane with
little distortion. This tutorial classifies points of a hierarchy embedded
in the Poincaré disk, the model of the hyperbolic plane used by
``HyperbolicPoincare``, and looks at where the prototypes go.

The data
--------

A tree with branching factor 3 is embedded in the disk: a node of depth
:math:`d` lies at hyperbolic distance :math:`1.2\,d` from the centre, in
the middle of its parent's angular sector. Each of the nine subtrees rooted
at depth 2 is a class, and samples are scattered around the nodes of
depths 2 to 4 with the exponential map of the disk:

.. code-block:: python

   import numpy as np
   import jax
   import jax.numpy as jnp
   from prosemble.core.manifolds import HyperbolicPoincare

   def tree_samples(per_node, seed, step=1.2, noise=1.0, branch=3, depth=4):
       nodes = []

       def add(d, lo, hi, cls):
           theta, r = (lo + hi) / 2, np.tanh(d * step / 2)
           if d >= 2:
               nodes.append((np.array([r * np.cos(theta), r * np.sin(theta)]), cls))
           if d < depth:
               w = (hi - lo) / branch
               for i in range(branch):
                   add(d + 1, lo + i * w, lo + (i + 1) * w,
                       cls if d >= 2 else cls * branch + i)

       for c in range(branch):
           add(1, c * 2 * np.pi / branch, (c + 1) * 2 * np.pi / branch, c)
       exp = jax.jit(HyperbolicPoincare(2).exp_map)
       rng = np.random.default_rng(seed)
       X, y = [], []
       for point, cls in nodes:
           lam = 2 / (1 - point @ point)        # conformal factor at the node
           for _ in range(per_node):
               v = noise * rng.normal(size=2) / lam
               X.append(np.asarray(exp(jnp.asarray(point, dtype=jnp.float32),
                                       jnp.asarray(v, dtype=jnp.float32))))
               y.append(cls)
       return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

   X_train, y_train = tree_samples(6, seed=6000)
   X_test, y_test = tree_samples(20, seed=99)

Most samples lie close to the edge of the disk, where the deeper nodes are:
the edge is infinitely far away in hyperbolic distance.

Train in the disk
-----------------

.. code-block:: python

   from prosemble.models import RiemannianSRNG

   model = RiemannianSRNG(manifold=HyperbolicPoincare(2), n_prototypes_per_class=1,
                          max_iter=300, lr=0.05, random_seed=0)
   model.fit(jnp.asarray(X_train), jnp.asarray(y_train))

Where the prototypes go
-----------------------

.. figure:: /_static/figures/tutorial-hyperbolic-light.png
   :class: only-light pr-figure

.. figure:: /_static/figures/tutorial-hyperbolic-dark.png
   :class: only-dark pr-figure

   Decision regions, test samples and prototypes of GLVQ on the disk
   coordinates (left) and of RiemannianSRNG with the hyperbolic distance
   (right). Accuracies are means over five training sets.

Both models classify about 98 % of the test points correctly (0.982 for
GLVQ on the coordinates, 0.980 for the hyperbolic model): the classes are
angular sectors of the disk, which the Euclidean distance separates as
well. The difference is in the prototypes.

* GLVQ places its prototypes at the rim (radius 0.97 on average), among
  the leaves of each subtree, 1.85 away from the subtree's root in
  hyperbolic distance.
* The hyperbolic prototypes stay at radius 0.89, on average 0.45 from the
  root of their subtree, less than half the length of one edge of the tree
  (1.2). Each prototype marks the common ancestor of its class.

The decision boundaries of the hyperbolic model are geodesics of the disk,
arcs that meet the edge at right angles; near the centre they bend, where
the Euclidean boundaries are straight lines.

When to use hyperbolic models
-----------------------------

On these data the hyperbolic distance does not improve accuracy, and a
Euclidean model on the disk coordinates is a reasonable baseline. The
hyperbolic model gives prototypes that can be read in the terms of the
hierarchy, and it is the consistent choice when the data were embedded
with the hyperbolic distance, for example by a hyperbolic embedding of a
taxonomy. Compare both on your data. :doc:`/user_guide/riemannian`
describes ``HyperbolicPoincare`` and the models that support it.
