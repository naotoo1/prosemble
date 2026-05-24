"""Reject option with calibrated uncertainty for prototype classifiers.

Implements Chow's optimal reject rule (1970) adapted for LVQ models.
When the confidence in a prediction is below a threshold, the model
abstains from classifying the sample (returns -1 or a reject label).

For LVQ, the confidence measure is the negative mu-ratio:

.. math::

    \\text{confidence}(x) = \\frac{d^-(x) - d^+(x)}{d^-(x) + d^+(x)}

This is the natural confidence measure from the GLVQ loss. Values near 0
mean the sample is on the decision boundary (uncertain), values near 1
mean high confidence.

References
----------
.. [1] Chow, C. K. (1970). On optimum recognition error and reject
       tradeoff. IEEE Transactions on Information Theory.
.. [2] Fischer, L., Hammer, B., & Wersing, H. (2015). Rejection
       strategies for learning vector quantization. Neurocomputing.
"""

import jax.numpy as jnp
import numpy as np


REJECT_LABEL = -1


class RejectOptionMixin:
    """Mixin providing reject option for any prototype-based classifier.

    Adds ``predict_with_rejection()`` and ``confidence()`` methods to
    any supervised model that has fitted prototypes and prototype labels.

    The reject decision is based on the relative margin:

    .. math::

        \\text{confidence}(x) = \\frac{d^-(x) - d^+(x)}{d^-(x) + d^+(x)}

    If confidence < threshold, the sample is rejected (label = -1).

    Methods
    -------
    confidence(X) -> array
        Compute confidence scores for each sample.
    predict_with_rejection(X, threshold) -> array
        Predict with rejection. Returns -1 for rejected samples.
    rejection_rate(X, threshold) -> float
        Compute the fraction of samples that would be rejected.
    optimal_threshold(X, y, cost_reject, cost_error) -> float
        Find the optimal rejection threshold minimizing total risk.
    """

    def confidence(self, X):
        """Compute confidence scores based on the relative margin.

        The confidence is the negative of the GLVQ mu-ratio:
        confidence(x) = (d_minus - d_plus) / (d_minus + d_plus)

        Values range from -1 (maximally wrong) to +1 (maximally confident).
        Values near 0 indicate the sample is on the decision boundary.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        scores : array of shape (n_samples,)
            Confidence scores in [-1, 1].
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        distances = self._compute_distances_for_rejection(X)
        proto_labels = self.prototype_labels_

        # Find closest same-class and closest different-class prototype
        # for each sample using the predicted label
        predictions = jnp.argmin(distances, axis=1)
        pred_labels = proto_labels[predictions]

        # d_plus: min distance to same-class prototype
        same_mask = (pred_labels[:, None] == proto_labels[None, :])
        INF = jnp.finfo(distances.dtype).max
        d_plus = jnp.min(jnp.where(same_mask, distances, INF), axis=1)

        # d_minus: min distance to different-class prototype
        d_minus = jnp.min(jnp.where(~same_mask, distances, INF), axis=1)

        # Confidence = (d_minus - d_plus) / (d_minus + d_plus + eps)
        confidence = (d_minus - d_plus) / (d_minus + d_plus + 1e-10)
        return confidence

    def predict_with_rejection(self, X, threshold=0.0):
        """Predict class labels with rejection option.

        Samples with confidence below the threshold are assigned the
        reject label (-1) instead of a class prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        threshold : float
            Rejection threshold. Samples with confidence < threshold
            are rejected. Default: 0.0 (reject uncertain samples).
            Reasonable range: [0.0, 0.5].

        Returns
        -------
        labels : array of shape (n_samples,)
            Predicted labels. Rejected samples have label -1.
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        # Get predictions and confidence
        predictions = self.predict(X)
        conf = self.confidence(X)

        # Apply rejection
        labels = jnp.where(conf >= threshold, predictions, REJECT_LABEL)
        return labels

    def rejection_rate(self, X, threshold=0.0):
        """Compute the fraction of samples that would be rejected.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        threshold : float
            Rejection threshold. Default: 0.0.

        Returns
        -------
        rate : float
            Fraction of rejected samples in [0, 1].
        """
        conf = self.confidence(X)
        return float(jnp.mean(conf < threshold))

    def accuracy_coverage_curve(self, X, y, n_thresholds=50):
        """Compute accuracy-coverage curve (reject curve).

        Shows the trade-off between classification accuracy and coverage
        (1 - rejection_rate) as the threshold varies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,)
            True labels.
        n_thresholds : int
            Number of threshold values to evaluate. Default: 50.

        Returns
        -------
        thresholds : array of shape (n_thresholds,)
            Threshold values evaluated.
        accuracies : array of shape (n_thresholds,)
            Accuracy on non-rejected samples at each threshold.
        coverages : array of shape (n_thresholds,)
            Fraction of non-rejected samples at each threshold.
        """
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.int32)

        conf = np.asarray(self.confidence(X))
        preds = np.asarray(self.predict(X))
        y_np = np.asarray(y)

        thresholds = np.linspace(-1.0, 1.0, n_thresholds)
        accuracies = np.zeros(n_thresholds)
        coverages = np.zeros(n_thresholds)

        for i, t in enumerate(thresholds):
            accepted = conf >= t
            n_accepted = np.sum(accepted)
            coverages[i] = n_accepted / len(y_np)
            if n_accepted > 0:
                accuracies[i] = np.mean(preds[accepted] == y_np[accepted])
            else:
                accuracies[i] = 1.0  # No errors if nothing is accepted

        return thresholds, accuracies, coverages

    def optimal_threshold(self, X, y, cost_reject=0.5, cost_error=1.0,
                          n_thresholds=100):
        """Find the optimal rejection threshold minimizing total risk.

        Minimizes Chow's risk:
            Risk = P(error|accepted) * coverage + cost_reject * (1 - coverage)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            True labels.
        cost_reject : float
            Cost of rejecting a sample (relative to cost of error).
            Default: 0.5 (rejection costs half of a misclassification).
        cost_error : float
            Cost of a misclassification. Default: 1.0.
        n_thresholds : int
            Number of thresholds to evaluate. Default: 100.

        Returns
        -------
        optimal_threshold : float
            The threshold that minimizes total risk.
        """
        thresholds, accuracies, coverages = self.accuracy_coverage_curve(
            X, y, n_thresholds
        )

        # Risk = error_rate * coverage + cost_reject * reject_rate
        error_rates = 1.0 - accuracies
        reject_rates = 1.0 - coverages
        risks = cost_error * error_rates * coverages + cost_reject * reject_rates

        best_idx = np.argmin(risks)
        return float(thresholds[best_idx])

    def _compute_distances_for_rejection(self, X):
        """Compute distance matrix for rejection confidence.

        By default uses the model's standard distance function.
        Models that compute distances differently (e.g., kernel distances)
        should override this method.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        distances : array of shape (n_samples, n_prototypes)
        """
        return self.distance_fn(X, self.prototypes_)
