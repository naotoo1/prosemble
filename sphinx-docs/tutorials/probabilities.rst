Calibrated probabilities and rejection
======================================

.. currentmodule:: prosemble.models

A classifier that says "90 % sure" should be right nine times in ten. This
tutorial checks whether a soft prototype model is, corrects it with
temperature calibration, and uses the calibrated probabilities to set aside
the samples the model is unsure about.

The data and the model
----------------------

A synthetic three-class problem with overlapping classes, so that the
model makes enough mistakes to measure its calibration:

.. code-block:: python

   import jax.numpy as jnp
   from sklearn.datasets import make_classification
   from prosemble.core.pipeline import StandardScaler
   from prosemble.core.utils import train_test_split_jax
   from prosemble.models import DKLMRSLVQ

   X, y = make_classification(
       n_samples=1500, n_features=10, n_informative=5, n_redundant=2,
       n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=0)
   X, y = jnp.asarray(X, dtype=jnp.float32), jnp.asarray(y)
   X_train, X_test, y_train, y_test = train_test_split_jax(
       X, y, test_size=0.4, random_seed=0)
   scaler = StandardScaler().fit(X_train)
   X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

   model = DKLMRSLVQ(n_prototypes_per_class=2, max_iter=200, lr=0.01,
                     random_seed=0)
   model.fit(X_train, y_train)

:class:`DKLMRSLVQ` is a soft model: each prototype scores a sample with
its own learned kernel metric, and ``predict_proba`` returns the class
posteriors that the model is trained on. It classifies 84.2 % of the test samples correctly.

Calibrate
---------

The fitted model is overconfident. Its mean confidence on the test data
(the largest class probability, averaged) is 0.93, while its accuracy is
0.84. ``calibrate`` fits a single temperature :math:`T` on data the model
was not trained on, here by 3-fold cross-fitting on the training data, and
divides the model's scores by it:

.. code-block:: python

   model.calibrate(X_train, y_train, cv=3)
   model.temperature_          # 2.56

.. figure:: /_static/figures/tutorial-calibration-light.png
   :class: only-light pr-figure

.. figure:: /_static/figures/tutorial-calibration-dark.png
   :class: only-dark pr-figure

   Left: accuracy against confidence on the test data, before and after
   calibration. Right: accuracy on the samples the calibrated model is
   most confident about.

After calibration the mean confidence is 0.80, and the reliability curve
follows the diagonal. Calibration does not change any predicted label.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Test data
     - Fitted
     - Calibrated
   * - Log-loss
     - 0.498
     - 0.399
   * - Expected calibration error
     - 0.089
     - 0.048
   * - Mean confidence (accuracy 0.842)
     - 0.929
     - 0.800

.. code-block:: python

   from prosemble.core.evaluation import expected_calibration_error

   proba = model.predict_proba(X_test)
   expected_calibration_error(proba, y_test)     # 0.048

Calibration minimises the log-loss of the held-out scores, so the log-loss
always benefits; the calibration error usually drops too, but not
necessarily.

Reject uncertain samples
------------------------

With calibrated probabilities, the confidence can decide which samples to
leave to a person. Keeping the 80 % of test samples with the highest
confidence raises the accuracy on them from 0.842 to 0.912:

.. code-block:: python

   from prosemble.core.evaluation import selective_accuracy

   selective_accuracy(proba, y_test, coverage=0.8)       # 0.912

   labels = model.predict_with_rejection(X_test, confidence=0.7)
   # -1 where the largest class probability is below 0.7

Next steps
----------

* :doc:`/user_guide/probabilities` covers calibration and both kinds of
  rejection in detail.
* On the 36-dataset :doc:`benchmark </benchmarks/index>`, calibrated
  :class:`DKLMRSLVQ` shared the lowest calibration error with a tuned SVM.
