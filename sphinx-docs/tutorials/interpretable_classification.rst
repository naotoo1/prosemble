Interpretable classification
============================

.. currentmodule:: prosemble.models

This tutorial trains a classifier on the wine data and reads off what it
learned: a typical wine of each cultivar, the chemical measurements that
tell the cultivars apart, and a two-dimensional view of the data as the
classifier sees it.

The data
--------

The wine data has 178 wines from three cultivars and 13 chemical
measurements per wine. Standardise the measurements so that their units do
not decide the distance:

.. code-block:: python

   import jax.numpy as jnp
   from sklearn.datasets import load_wine
   from prosemble.core.pipeline import StandardScaler
   from prosemble.core.utils import train_test_split_jax

   wine = load_wine()
   X = jnp.asarray(wine.data, dtype=jnp.float32)
   y = jnp.asarray(wine.target)
   X_train, X_test, y_train, y_test = train_test_split_jax(
       X, y, test_size=0.3, random_seed=0)

   scaler = StandardScaler().fit(X_train)
   X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

Train
-----

:class:`GMLVQ` learns one prototype per cultivar and a matrix
:math:`\Omega` that defines the distance
:math:`d(x, w) = (x - w)^\top \Lambda (x - w)` with
:math:`\Lambda = \Omega^\top \Omega`:

.. code-block:: python

   from prosemble.models import GMLVQ

   model = GMLVQ(n_prototypes_per_class=1, max_iter=300, lr=0.01,
                 random_seed=0)
   model.fit(X_train, y_train)

   accuracy = jnp.mean(model.predict(X_test) == y_test)    # 0.981

The model classifies 53 of the 54 test wines correctly. Plain
:class:`GLVQ`, which uses the Euclidean distance, classifies all 54
correctly: this dataset is easy, and the metric is not needed for accuracy.
What it adds is an explanation.

What the model learned
----------------------

.. figure:: /_static/figures/tutorial-wine-light.png
   :class: only-light pr-figure

.. figure:: /_static/figures/tutorial-wine-dark.png
   :class: only-dark pr-figure

   Left: relevance of each measurement. Right: all wines projected on the
   two directions the learned distance uses, with the three prototypes.

**Relevances.** The diagonal of :math:`\Lambda` says how much each
measurement counts in the distance:

.. code-block:: python

   import numpy as np

   lam = np.asarray(model.lambda_matrix)
   relevance = np.diag(lam) / np.trace(lam)

Flavanoids (0.23), proline (0.14) and colour intensity (0.11) carry the
most weight, and magnesium and non-flavanoid phenols the least. A
practitioner can check such a ranking against domain knowledge, which is
much harder to do with a model that does not expose its metric.

**The model's view of the data.** :math:`\Lambda` has, in effect, rank two:
its two largest eigenvalues hold 54 % and 46 % of its trace, and the other
eleven together less than 1 %. The classifier therefore compares wines in a
two-dimensional subspace, and projecting on it loses nothing of what the
model uses:

.. code-block:: python

   eigval, eigvec = np.linalg.eigh(lam)
   top = eigvec[:, ::-1][:, :2] * np.sqrt(eigval[::-1][:2])
   Z = np.asarray(X_train) @ top                    # 2-D view of the wines
   W = np.asarray(model.prototypes_) @ top          # and of the prototypes

In this view the three cultivars form three separate groups, each around
its prototype.

**The prototypes.** ``model.prototypes_`` holds one typical wine per
cultivar. ``scaler`` standardised the data, so multiply by the standard
deviations and add the means of the training data to read them in the
original units.

Next steps
----------

* :doc:`/user_guide/classification/metric_learning` describes the other
  metric-learning models, including local metrics (:class:`LGMLVQ`).
* :doc:`/user_guide/interpretation` collects the ways to inspect a fitted
  model.
