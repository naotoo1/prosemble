Probabilities, calibration and rejection
========================================

.. currentmodule:: prosemble.models

Every classifier returns class probabilities from ``predict_proba``. How
they are obtained differs between families, and so does how far they can
be trusted:

* The GLVQ-type models (GLVQ, metric learning, Neural Gas, ...) apply a
  softmax to the negative distance to the nearest prototype of each class.
  These scores rank samples sensibly but are not probabilities in a
  statistical sense; on the classification benchmark their log-loss is far
  above that of the probabilistic models.
* The probabilistic models (:doc:`classification/probabilistic`) return
  the posterior of a Gaussian mixture, trained by maximum likelihood.
  After calibration they are among the best-calibrated models on the
  benchmark (:doc:`/benchmarks/index`).

When probabilities matter, use a probabilistic model and calibrate it.

Calibration
-----------

The soft models learn their width
:math:`\sigma` on training data they fit closely, so their class
posteriors tend to be too confident.  ``calibrate`` fits a temperature
:math:`T`, dividing every component score by it, that minimises the
negative log-likelihood of data the model was not fitted to.  With
``cv=k`` it fits ``k`` copies of the model on folds of the training data
and uses their scores on the left-out folds; the fitted model is kept.
Calibration changes ``predict_proba`` and ``predict_with_rejection``,
never the labels returned by ``predict``.

.. code-block:: python

   from prosemble.models import WLMRSLVQ

   model = WLMRSLVQ(n_prototypes_per_class=2).fit(X_train, y_train)
   model.calibrate(X_train, y_train, cv=3)     # or calibrate(X_val, y_val)
   proba = model.predict_proba(X_test)
   print(model.temperature_)

On 13 tabular datasets cross-fitted calibration lowered the mean
log-loss of ``DKLMRSLVQ`` from 0.39 to 0.28 and of ``WLMRSLVQ`` from 0.72
to 0.37 (an RBF SVM with Platt scaling: 0.29).  It costs ``cv`` extra
fits.

``calibrate`` is available on :class:`DKRSLVQ`, :class:`DKMRSLVQ`,
:class:`DKLMRSLVQ`, :class:`WRSLVQ`, :class:`WMRSLVQ`, :class:`WLMRSLVQ`,
:class:`RiemannianRSLVQ`, :class:`RiemannianMRSLVQ` and the one-class
models :class:`OCDKRSLVQ`, :class:`OCWRSLVQ` and
:class:`OCRiemannianRSLVQ`.

Rejecting uncertain samples
---------------------------

The ``RejectOptionMixin`` adds calibrated rejection to any supervised
prototype classifier. When the model is uncertain about a prediction
(the sample lies near the decision boundary), it abstains instead of
guessing.

The confidence measure is the GLVQ relative margin:

.. math::

   \text{confidence}(x) = \frac{d^-(x) - d^+(x)}{d^-(x) + d^+(x)}

Values near 0 indicate the decision boundary; values near 1 indicate
high confidence.

.. code-block:: python

   from prosemble.models import GLVQ

   model = GLVQ(n_prototypes_per_class=2, max_iter=100, lr=0.01)
   model.fit(X_train, y_train)

   # Predict with rejection (uncertain samples get label -1)
   labels = model.predict_with_rejection(X_test, threshold=0.1)
   rejected = labels == -1
   print(f"Rejected {rejected.sum()} of {len(labels)} samples")

   # Compute confidence scores
   conf = model.confidence(X_test)

   # Find the optimal threshold minimizing Chow's risk
   threshold = model.optimal_threshold(
       X_val, y_val,
       cost_reject=0.5,    # rejection costs half of a misclassification
       cost_error=1.0,
   )

   # Accuracy-coverage curve
   thresholds, accuracies, coverages = model.accuracy_coverage_curve(X_val, y_val)

The probabilistic models reject by the posterior instead: a sample is
rejected when its largest class probability is below ``confidence``, or
below the ``rejection_confidence`` given to the constructor.

.. code-block:: python

   from prosemble.models import DKLMRSLVQ

   model = DKLMRSLVQ(n_prototypes_per_class=2, max_iter=200, lr=0.01)
   model.fit(X_train, y_train)
   model.calibrate(X_train, y_train, cv=3)

   labels = model.predict_with_rejection(X_test, confidence=0.8)

Measuring probability quality
-----------------------------

:mod:`prosemble.core.evaluation` provides two measures of predicted
probabilities. The expected calibration error compares the confidence of
predictions with their accuracy; selective accuracy is the accuracy on the
fraction of samples the model is most confident about, which shows what
rejecting the rest would gain.

.. code-block:: python

   from prosemble.core.evaluation import (
       expected_calibration_error, selective_accuracy)

   proba = model.predict_proba(X_test)
   ece = expected_calibration_error(proba, y_test)
   acc80 = selective_accuracy(proba, y_test, coverage=0.8)

See :doc:`/tutorials/probabilities` for a complete example.
