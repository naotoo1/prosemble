Prototype classifiers
=====================

.. currentmodule:: prosemble.models

Learning vector quantization (LVQ) represents every class by one or more
**prototypes**, points in the input space, and assigns a sample the label of
its nearest prototype. Generalized LVQ (:class:`GLVQ`) trains the prototypes
by gradient descent on the relative distance cost

.. math::

   \mu(x) = \frac{d^+(x) - d^-(x)}{d^+(x) + d^-(x)},

where :math:`d^+` is the distance to the closest prototype of the correct
class and :math:`d^-` the distance to the closest prototype of any other
class. :math:`\mu` is negative exactly when the sample is classified
correctly, and its magnitude is a margin.

The models on this page all use the squared Euclidean distance. They are
the baselines that the other families extend with a learned metric
(:doc:`metric_learning`), a probabilistic cost (:doc:`probabilistic`) or
neighbourhood cooperation (:doc:`neural_gas`).

GLVQ — Baseline
---------------

Generalized Learning Vector Quantization is the foundation model. It
learns prototype positions to minimize the relative distance cost
:math:`\mu = (d^+ - d^-) / (d^+ + d^-)`.

.. code-block:: python

   from prosemble.models import GLVQ
   from prosemble.datasets import load_iris_jax
   from prosemble.core.utils import train_test_split_jax

   dataset = load_iris_jax()
   X, y = dataset.input_data, dataset.labels
   X_train, X_test, y_train, y_test = train_test_split_jax(X, y)

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X_train, y_train)

   predictions = model.predict(X_test)
   probabilities = model.predict_proba(X_test)

**Unequal prototypes per class:**

.. code-block:: python

   model = GLVQ(
       n_prototypes_per_class={0: 3, 1: 2, 2: 1},
       prototypes_initializer='class_conditional_mean',
   )

**Prototype initializers** (pass as string):

- ``'stratified_random'`` — random samples per class (default)
- ``'class_mean'`` — class centroids
- ``'class_conditional_mean'`` — class centroids replicated
- ``'stratified_noise'`` — random samples + Gaussian noise
- ``'random_normal'`` — random normal initialization
- ``'uniform'`` — random uniform
- ``'zeros'`` — zero initialization
- ``'ones'`` — ones initialization
- ``'fill_value'`` — constant fill value (pass ``value=`` kwarg)

For unsupervised models, ``selection_init`` and ``mean_init`` from
``prosemble.core.initializers`` provide classless alternatives.

You can also pass a callable for custom initialization (e.g., ``literal_init(values)``
from ``prosemble.core.initializers``).

CELVQ — Cross-Entropy
---------------------

Replaces the GLVQ :math:`\mu` cost with cross-entropy loss over class
probabilities derived from distances. Produces calibrated probabilities.

.. code-block:: python

   from prosemble.models import CELVQ

   model = CELVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)
   proba = model.predict_proba(X_test)  # calibrated class probabilities

LVQ1 and LVQ2.1 — Classic Non-Gradient
--------------------------------------

Simple competitive learning without explicit gradients. LVQ1 updates only
the nearest prototype; LVQ2.1 updates both nearest correct and incorrect.

.. code-block:: python

   from prosemble.models import LVQ1, LVQ21

   model = LVQ1(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
   )
   model.fit(X_train, y_train)

Median LVQ
----------

Constrains prototypes to be actual data points for maximum interpretability.

.. code-block:: python

   from prosemble.models import MedianLVQ

   model = MedianLVQ(
       n_prototypes_per_class=1,
       max_iter=50,
       random_seed=42,
   )
   model.fit(X_train, y_train)
   # model.prototypes_ are actual training examples
