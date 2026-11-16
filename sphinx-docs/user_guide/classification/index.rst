Classification
==============

.. currentmodule:: prosemble.models

A prototype classifier stores a few labelled points, the **prototypes**,
and labels a sample by comparing it with them. The families below differ
in how they compare (the distance) and in the cost they are trained with.
All of them share one interface:

.. code-block:: python

   from prosemble.models import GMLVQ

   model = GMLVQ(n_prototypes_per_class=2, max_iter=200, lr=0.01)
   model.fit(X_train, y_train)

   model.predict(X_test)          # labels
   model.predict_proba(X_test)    # class probabilities
   model.prototypes_              # (n_prototypes, n_features)

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`dot-fill` Prototype classifiers
      :link: lvq
      :link-type: doc

      GLVQ, cross-entropy LVQ, LVQ1 and LVQ2.1, median LVQ: the
      Euclidean baselines.

   .. grid-item-card:: :octicon:`telescope` Metric learning
      :link: metric_learning
      :link-type: doc

      GRLVQ, GMLVQ, LGMLVQ, GTLVQ: learn which features matter.

   .. grid-item-card:: :octicon:`graph` Probabilistic classifiers
      :link: probabilistic
      :link-type: doc

      SLVQ, RSLVQ and their matrix variants: Gaussian mixtures with
      posterior class probabilities.

   .. grid-item-card:: :octicon:`git-merge` Neural Gas classifiers
      :link: neural_gas
      :link-type: doc

      SNG, SRNG, SMNG and the cross-entropy and soft families: rank-based
      cooperation between prototypes.

   .. grid-item-card:: :octicon:`ellipsis` Distributional prototypes
      :link: wasserstein
      :link-type: doc

      Wasserstein LVQ: Gaussian prototypes with a learned spread.

   .. grid-item-card:: :octicon:`stack` Deep, Siamese and image models
      :link: deep
      :link-type: doc

      A neural network before the prototype layer; classification by
      components.

The same ideas extend beyond Euclidean vectors: see
:doc:`/user_guide/kernels`, :doc:`/user_guide/riemannian` and
:doc:`/user_guide/one_class`. To pick a model, start with
:doc:`/getting_started/choosing_a_model`.

.. toctree::
   :hidden:

   lvq
   metric_learning
   probabilistic
   neural_gas
   wasserstein
   deep

Common interface
----------------

Every classifier has these methods:

.. list-table::
   :widths: 40 60

   * - ``fit(X, y)``
     - Train. Optional ``validation_data``, ``sample_weight``, and
       ``resume=True`` to continue from the fitted state.
   * - ``partial_fit(X, y)``
     - One more update on new data (online learning).
   * - ``predict(X)``
     - Class labels.
   * - ``predict_proba(X)``
     - Class probabilities; columns follow ``classes_``.
   * - ``predict_with_rejection(X, ...)``
     - Labels, or ``-1`` where the model abstains
       (:doc:`/user_guide/probabilities`).
   * - ``save(path)``, ``Model.load(path)``
     - Persistence (:doc:`/user_guide/deployment`).
   * - ``get_params()``, ``set_params(**params)``
     - Hyperparameters, as used by ``clone`` and ``GridSearchCV``
       (:doc:`/user_guide/model_selection`).

and these fitted attributes:

.. list-table::
   :widths: 40 60

   * - ``prototypes_``
     - Prototype positions, one row per prototype.
   * - ``prototype_labels_``
     - The class of each prototype.
   * - ``classes_``
     - The class labels seen in ``fit``.
   * - ``n_iter_``, ``loss_``, ``loss_history_``
     - Iterations run, final loss, and the loss per iteration.

Labels can be any values; they are encoded internally and ``predict``
returns the original values.
