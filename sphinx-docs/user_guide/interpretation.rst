Interpreting models
===================

.. currentmodule:: prosemble.models

A prototype model's decision can be traced to a few learned objects: the
prototypes a sample was compared with, the metric used to compare them,
and, for the soft models, the width of each mixture component. This page
shows how to read them. :doc:`/tutorials/interpretable_classification`
works through a complete example.

Prototypes
----------

Except in :class:`LVQMLN` and :class:`PLVQ`, prototypes live in the input
space, so each one can be read like a data point: a typical example of its
class. If the data were standardised, transform the prototypes back before
reporting them in the original units.

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=2, max_iter=200, lr=0.01)
   model.fit(X_train, y_train)

   for w, label in zip(model.prototypes_, model.prototype_labels_):
       print(label, w)

The confidence of a decision is the relative margin between the nearest
prototype of the predicted class and the nearest prototype of any other
class, ``model.confidence(X)``; see :doc:`probabilities`.

Feature relevances and learned metrics
--------------------------------------

Metric-learning models (:doc:`classification/metric_learning`) learn how to
weigh the features:

* :class:`GRLVQ` and :class:`SRNG` learn one relevance per feature,
  ``relevances_``. The relevances are non-negative and sum to one; large
  values mark the features the classification relies on.
* :class:`GMLVQ` learns a matrix :math:`\Omega` and the distance
  :math:`(x - w)^\top \Lambda (x - w)` with :math:`\Lambda = \Omega^\top
  \Omega` (``lambda_matrix``). The diagonal of :math:`\Lambda` is the
  relevance of each feature; off-diagonal entries show pairs of features
  that act together.
* :class:`LGMLVQ` learns one matrix per prototype (``omegas_``), so
  relevances can differ between regions of the input space.

.. code-block:: python

   import jax.numpy as jnp
   from prosemble.models import GMLVQ

   model = GMLVQ(n_prototypes_per_class=1, max_iter=300, lr=0.01)
   model.fit(X_train, y_train)

   relevance = jnp.diag(model.lambda_matrix)   # one value per feature
   ranking = jnp.argsort(-relevance)           # most relevant first

The eigenvectors of :math:`\Lambda` with the largest eigenvalues span the
directions the classifier uses. Projecting the data on the leading two
gives a two-dimensional view of the classes as the model sees them.

Prototype Win Ratios
--------------------

Analyze how often each prototype wins on correctly classified samples.
This helps identify "dead" prototypes that never win and may be
candidates for removal or reinitialization:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=3, max_iter=100, lr=0.01)
   model.fit(X_train, y_train)

   ratios = model.prototype_win_ratios(X_train, y_train)

   for i, r in enumerate(ratios):
       label = model.prototype_labels_[i]
       print(f"Prototype {i} (class {label}): win ratio {r:.3f}")

A win ratio of 0.0 means the prototype never won on any correctly
classified sample and is effectively unused.

Live Visualization
------------------

Enable real-time training visualization for clustering models:

.. code-block:: python

   from prosemble.models import FCM

   model = FCM(
       n_clusters=3,
       max_iter=100,
       plot_steps=True,
   )
   model.fit(X)

Geodesic Interpolation
----------------------

For Riemannian models, geodesic utilities enable visualization and
analysis of decision boundaries on curved manifolds.

.. code-block:: python

   from prosemble.core.geodesic import (
       geodesic_interpolation,
       geodesic_midpoint,
       decision_boundary_point,
       prototype_geodesic_distances,
       inter_class_geodesics,
   )
   from prosemble.core.manifolds import SO

   manifold = SO(3)

   # Compute geodesic path between two SO(3) prototypes
   path = geodesic_interpolation(manifold, proto_a, proto_b, n_points=50)

   # Find the decision boundary along the geodesic
   boundary_pt, t_boundary = decision_boundary_point(
       manifold, proto_a, proto_b,
   )
   print(f"Boundary at t={t_boundary:.3f}")  # t=0.5 for symmetric manifolds

   # Pairwise geodesic distance matrix between all prototypes
   dist_matrix = prototype_geodesic_distances(
       manifold, prototypes, proto_labels,
   )

   # All inter-class geodesics with boundary locations
   geodesics = inter_class_geodesics(
       manifold, prototypes, proto_labels,
   )
