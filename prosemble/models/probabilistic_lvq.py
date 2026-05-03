"""
Probabilistic LVQ models: SLVQ and RSLVQ.

Soft LVQ (SLVQ) and Robust Soft LVQ (RSLVQ) use Gaussian mixture
models to define class-conditional probabilities and optimize
likelihood-based objectives.

References
----------
.. [1] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation.
.. [2] Seo, S., & Obermayer, K. (2007). Soft Nearest Prototype
       Classification. IEEE Trans. Neural Networks.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.losses import nllr_loss, rslvq_loss


class SLVQ(SupervisedPrototypeModel):
    """Soft Learning Vector Quantization.

    Uses Gaussian mixture probabilities:
        p(k|x) = exp(-d²/2σ²) / Σ exp(-d²/2σ²)
        P(class|x) = Σ_{k∈class} p(k|x)
    Loss: -log(P(correct) / P(wrong))

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture.
    rejection_confidence : float, optional
        Minimum class probability for a confident prediction (0 to 1).
        Samples below this threshold are rejected (labeled -1) when using
        ``predict_with_rejection()``. Default is None (no rejection).
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    """

    def __init__(self, sigma=1.0, rejection_confidence=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rejection_confidence = rejection_confidence

    def _compute_loss(self, params, X, y, proto_labels):
        distances = self.distance_fn(X, params['prototypes'])
        return nllr_loss(distances, y, proto_labels, sigma=self.sigma)

    def predict_with_rejection(self, X, confidence=None):
        """Predict with rejection option.

        Samples whose maximum class probability is below the confidence
        threshold are assigned label -1 (rejected / "I don't know").

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        confidence : float, optional
            Override the model's rejection_confidence for this call.

        Returns
        -------
        labels : array of shape (n_samples,)
            Predicted labels, or -1 for rejected samples.
        """
        self._check_fitted()
        threshold = confidence if confidence is not None else self.rejection_confidence
        if threshold is None:
            return self.predict(X)

        X = jnp.asarray(X, dtype=jnp.float32)
        proba = self.predict_proba(X)
        max_proba = jnp.max(proba, axis=1)
        preds = jnp.argmax(proba, axis=1)
        return jnp.where(max_proba >= threshold, preds, -1)

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma'] = self.sigma
        hp['rejection_confidence'] = self.rejection_confidence
        return hp


class RSLVQ(SupervisedPrototypeModel):
    """Robust Soft Learning Vector Quantization.

    Like SLVQ but with a more robust denominator:
    Loss: -log(P(correct) / P(all))

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture.
    rejection_confidence : float, optional
        Minimum class probability for a confident prediction (0 to 1).
        Samples below this threshold are rejected (labeled -1) when using
        ``predict_with_rejection()``. Default is None (no rejection).
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    """

    def __init__(self, sigma=1.0, rejection_confidence=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rejection_confidence = rejection_confidence

    def _compute_loss(self, params, X, y, proto_labels):
        distances = self.distance_fn(X, params['prototypes'])
        return rslvq_loss(distances, y, proto_labels, sigma=self.sigma)

    def predict_with_rejection(self, X, confidence=None):
        """Predict with rejection option.

        Samples whose maximum class probability is below the confidence
        threshold are assigned label -1 (rejected / "I don't know").

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        confidence : float, optional
            Override the model's rejection_confidence for this call.

        Returns
        -------
        labels : array of shape (n_samples,)
            Predicted labels, or -1 for rejected samples.
        """
        self._check_fitted()
        threshold = confidence if confidence is not None else self.rejection_confidence
        if threshold is None:
            return self.predict(X)

        X = jnp.asarray(X, dtype=jnp.float32)
        proba = self.predict_proba(X)
        max_proba = jnp.max(proba, axis=1)
        preds = jnp.argmax(proba, axis=1)
        return jnp.where(max_proba >= threshold, preds, -1)

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma'] = self.sigma
        hp['rejection_confidence'] = self.rejection_confidence
        return hp
