Quick Start
===========

This guide walks through a complete example: training a GLVQ classifier on the
Iris dataset, making predictions, and inspecting the results.

Loading Data
------------

Prosemble provides built-in dataset loaders that return JAX arrays:

.. code-block:: python

   from prosemble.datasets import load_iris_jax
   from prosemble.core.utils import train_test_split_jax

   dataset = load_iris_jax()
   X, y = dataset.input_data, dataset.labels

   X_train, X_test, y_train, y_test = train_test_split_jax(
       X, y, test_size=0.2, random_seed=42
   )

Training a Model
----------------

All supervised models follow the same ``fit`` / ``predict`` API:

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(
       n_prototypes_per_class=2,
       max_iter=100,
       lr=0.01,
       random_seed=42,
   )
   model.fit(X_train, y_train)

   print(f"Converged in {model.n_iter_} iterations")
   print(f"Final loss: {model.loss_:.4f}")

Making Predictions
------------------

.. code-block:: python

   import jax.numpy as jnp

   predictions = model.predict(X_test)
   accuracy = float(jnp.mean(predictions == y_test))
   print(f"Test accuracy: {accuracy:.2%}")

   # Class probabilities
   probabilities = model.predict_proba(X_test)

Inspecting the Model
--------------------

After training, fitted attributes are available:

.. code-block:: python

   # Prototype positions and labels
   print(model.prototypes_.shape)       # (n_prototypes, n_features)
   print(model.prototype_labels_)       # class label per prototype

   # Training history
   print(model.loss_history_[-5:])      # last 5 loss values

Unsupervised Models
-------------------

Clustering models use the same pattern, without labels:

.. code-block:: python

   from prosemble.models import FCM

   fcm = FCM(n_clusters=3, fuzzifier=2.0, max_iter=100)
   fcm.fit(X_train)

   labels = fcm.predict(X_test)
   membership = fcm.predict_proba(X_test)  # fuzzy membership matrix
   print(fcm.centroids_.shape)             # (n_clusters, n_features)

Next Steps
----------

- :doc:`guides/supervised` — Full guide to supervised LVQ models
- :doc:`guides/clustering` — Fuzzy clustering models
- :doc:`guides/topology` — Neural Gas, SOM, and topology models
- :doc:`guides/one_class` — One-class classification
- :doc:`guides/advanced` — JIT compilation, mixed precision, checkpointing
