Quickstart
==========

.. currentmodule:: prosemble.models

This page trains a classifier on the Iris data, evaluates it, and looks
inside it. It takes a few seconds on a laptop CPU.

Load and prepare the data
-------------------------

Prosemble works on JAX arrays and ships a few small datasets. Distances
depend on the units of the features, so standardise them first:

.. code-block:: python

   from prosemble.datasets import load_iris_jax
   from prosemble.core.utils import train_test_split_jax
   from prosemble.core.pipeline import StandardScaler

   data = load_iris_jax()
   X_train, X_test, y_train, y_test = train_test_split_jax(
       data.input_data, data.labels, test_size=0.3, random_seed=0)

   scaler = StandardScaler().fit(X_train)
   X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

Train a model
-------------

:class:`GMLVQ` learns one prototype per class together with a distance
that weighs the features:

.. code-block:: python

   from prosemble.models import GMLVQ

   model = GMLVQ(n_prototypes_per_class=1, max_iter=200, lr=0.05,
                 random_seed=0)
   model.fit(X_train, y_train)

Training is compiled with JAX on the first call; later calls with data of
the same shape reuse the compiled program.

Predict
-------

.. code-block:: python

   import jax.numpy as jnp

   pred = model.predict(X_test)
   print(f"Test accuracy: {float(jnp.mean(pred == y_test)):.3f}")
   # Test accuracy: 0.978

   model.predict_proba(X_test[:3])
   # [[1.   0.   0.  ]
   #  [1.   0.   0.  ]
   #  [0.   0.06 0.94]]

Look inside
-----------

The model is three prototypes and a metric, and both can be read directly.
Each prototype is a typical flower of its class, in the standardised
units of the data:

.. code-block:: python

   model.prototypes_          # shape (3, 4): one row per prototype
   model.prototype_labels_    # [0 1 2]

The diagonal of the learned matrix :math:`\Lambda` (``lambda_matrix``)
says how much each feature counts in the distance:

.. code-block:: python

   relevance = jnp.diag(model.lambda_matrix)
   relevance = relevance / relevance.sum()

   names = ["sepal length", "sepal width", "petal length", "petal width"]
   for name, r in zip(names, relevance):
       print(f"{name:13s} {float(r):.2f}")
   # sepal length  0.10
   # sepal width   0.21
   # petal length  0.41
   # petal width   0.28

The petal measurements carry most of the decision, as is well known for
this dataset. Training history is kept too:

.. code-block:: python

   model.n_iter_          # iterations run
   model.loss_history_    # the cost after each iteration

Save and load
-------------

.. code-block:: python

   model.save("gmlvq_iris.npz")
   loaded = GMLVQ.load("gmlvq_iris.npz")

Next steps
----------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`light-bulb` Concepts
      :link: concepts
      :link-type: doc

      What prototypes, metrics and soft models are.

   .. grid-item-card:: :octicon:`checklist` Choosing a model
      :link: choosing_a_model
      :link-type: doc

      Which model fits your task.

   .. grid-item-card:: :octicon:`book` User guide
      :link: /user_guide/index
      :link-type: doc

      Every model family and feature in depth.

   .. grid-item-card:: :octicon:`mortar-board` Tutorials
      :link: /tutorials/index
      :link-type: doc

      Complete worked examples with figures.
