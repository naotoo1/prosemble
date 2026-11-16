Model selection and evaluation
==============================

.. currentmodule:: prosemble.core

Prosemble includes the tools to preprocess data, estimate a model's
accuracy, tune its hyperparameters and compare models over many datasets.
They work on JAX arrays and follow the conventions of scikit-learn.

Pipelines
---------

A :class:`~pipeline.Pipeline` chains preprocessing steps with a final model,
so that the preprocessing is fitted on training data only. Prosemble
provides :class:`~pipeline.StandardScaler`, :class:`~pipeline.MinMaxScaler`
and :class:`~pipeline.PCA`.

.. code-block:: python

   from prosemble.core.pipeline import Pipeline, StandardScaler
   from prosemble.models import GMLVQ

   pipe = Pipeline([
       ("scale", StandardScaler()),
       ("model", GMLVQ(n_prototypes_per_class=1, max_iter=100, lr=0.01)),
   ])
   pipe.fit(X_train, y_train)
   pipe.predict(X_test)

Cross-validation
----------------

:func:`~model_selection.cross_val_score` fits a fresh copy of the model on
each of ``cv`` folds and returns the score on each held-out fold:

.. code-block:: python

   from prosemble.core.model_selection import cross_val_score
   from prosemble.datasets import load_breast_cancer_jax

   data = load_breast_cancer_jax()
   scores = cross_val_score(pipe, data.input_data, data.labels, cv=5)
   print(scores.mean())    # 0.970

``scoring`` accepts ``'accuracy'`` or any callable
``scorer(y_true, y_pred)``.

Hyperparameter search
---------------------

:class:`~model_selection.GridSearchCV` tries every combination in a grid
by cross-validation and refits the best one on all the data. Parameters of
a pipeline step are addressed as ``step__parameter``:

.. code-block:: python

   from prosemble.core.model_selection import GridSearchCV

   search = GridSearchCV(
       pipe,
       {"model__n_prototypes_per_class": [1, 2],
        "model__lr": [0.01, 0.05]},
       cv=3,
   )
   search.fit(data.input_data, data.labels)

   search.best_params_      # {'model__lr': 0.05, 'model__n_prototypes_per_class': 2}
   search.best_score_       # mean cross-validated accuracy of the best setting
   search.predict(X_new)    # uses the refitted best model

:func:`~model_selection.clone` makes an unfitted copy of a model, optionally
with new parameters: ``clone(model, lr=0.05)``.

Comparing models over many datasets
-----------------------------------

A difference in mean accuracy over a few datasets says little on its own.
:mod:`prosemble.core.evaluation` implements the procedure of Demšar (2006)
for comparing classifiers over many datasets. It starts from a table of
scores, one row per dataset and one column per method:

.. code-block:: python

   from prosemble.core.evaluation import (
       compare_to_reference, friedman_test, nemenyi_critical_difference)

   # table[i, j]: accuracy of method j on dataset i
   result = friedman_test(table)
   result.mean_ranks        # rank 1 is best
   result.p_value           # are the methods' ranks different at all?

   # Each method against a reference, Wilcoxon signed-rank tests with
   # Holm's correction for the number of comparisons
   for r in compare_to_reference(table, reference=0):
       print(r.method, r.wins, r.ties, r.losses, r.holm_p_value)

   # The smallest difference in mean ranks that the Nemenyi test calls
   # significant
   cd = nemenyi_critical_difference(n_methods=table.shape[1],
                                    n_datasets=table.shape[0])

The :doc:`benchmarks </benchmarks/index>` use exactly this protocol, and
:doc:`probabilities` describes the module's measures of predicted
probabilities.

.. rubric:: Reference

Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple
Data Sets. *Journal of Machine Learning Research*, 7, 1–30.
