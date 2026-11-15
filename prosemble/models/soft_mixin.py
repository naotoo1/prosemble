"""
Shared machinery for soft (probabilistic) prototype classifiers.

A soft prototype model is a labelled mixture: prototype :math:`k` carries
a class label :math:`c(k)` and a score

.. math::

    f_k(x) = \\log P(k) + \\log p(x | k) + \\text{const},

and is trained by maximising the likelihood ratio
:math:`\\sum_i \\log [p(x_i, y_i) / p(x_i)] = \\sum_i \\log P(y_i | x_i)`,
the robust soft LVQ cost. :class:`SoftAssignmentMixin` implements that
cost, a learnable global width, and prediction from the class posterior.
A model only has to say what its component scores are.
"""

import jax.numpy as jnp

from prosemble.core.losses import likelihood_ratio_loss, class_log_posterior


class SoftAssignmentMixin:
    """Likelihood-ratio training and posterior prediction for labelled mixtures.

    Place it before the prototype base class in the bases. Subclasses
    implement :meth:`_logits` and :meth:`_fitted_params`, and call
    :meth:`_setup_soft` from ``__init__``.

    The width :math:`\\sigma` of distance-based scores
    :math:`f_k = -d_k / (2 \\sigma^2)` is one global value. It is stored
    as ``params['log_sigma']`` so that it stays positive, and is learned
    by gradient descent on the same cost unless ``adapt_sigma`` is False.
    With one shared width the normalising constant of the components
    cancels in the likelihood ratio, so it does not have to be known.
    """

    _hyperparams = ('sigma', 'adapt_sigma', 'rejection_confidence')
    _fitted_array_names = ('sigma_',)

    def _setup_soft(self, sigma, adapt_sigma, rejection_confidence):
        if not (sigma == 'auto' or (isinstance(sigma, (int, float)) and sigma > 0)):
            raise ValueError(f"sigma must be 'auto' or a positive number, got {sigma!r}")
        self.sigma = sigma
        self.adapt_sigma = adapt_sigma
        self.rejection_confidence = rejection_confidence
        self.sigma_ = None
        if not adapt_sigma:
            frozen = list(self.freeze_params or [])
            if 'log_sigma' not in frozen:
                frozen.append('log_sigma')
            self.freeze_params = frozen

    # ------------------------------------------------------------------
    # To be provided by the model
    # ------------------------------------------------------------------

    def _logits(self, params, X):
        """Component scores :math:`f_k(x_i)`, array of shape (n, p)."""
        raise NotImplementedError

    def _fitted_params(self):
        """Parameter dict of the fitted model, as :meth:`_logits` expects it."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Width
    # ------------------------------------------------------------------

    def _initial_log_sigma(self, distances, y, proto_labels):
        """Initial :math:`\\log \\sigma` from squared distances at initialisation.

        ``sigma='auto'`` sets :math:`\\sigma^2` to the mean squared
        distance from a sample to its closest prototype of the same
        class, so that typical scores are of order one whatever the scale
        of the distance.
        """
        if self.sigma != 'auto':
            return jnp.log(jnp.asarray(float(self.sigma), dtype=jnp.float32))
        same_class = y[:, None] == proto_labels[None, :]
        own = jnp.min(jnp.where(same_class, distances, jnp.inf), axis=1)
        sigma2 = jnp.maximum(jnp.mean(own), 1e-6)
        return 0.5 * jnp.log(sigma2).astype(jnp.float32)

    @staticmethod
    def _scale_by_sigma(distances, log_sigma):
        """:math:`-d / (2 \\sigma^2)` for squared distances ``d``."""
        return -0.5 * distances * jnp.exp(-2.0 * log_sigma)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _compute_loss(self, params, X, y, proto_labels):
        return likelihood_ratio_loss(self._logits(params, X), y, proto_labels)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _class_log_posterior(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        logits = self._logits(self._fitted_params(), X)
        return class_log_posterior(logits, self.prototype_labels_, self.n_classes_)

    def predict(self, X):
        """Predict the class with the largest posterior :math:`P(c | x)`.

        With several prototypes per class this sums over the prototypes
        of a class and can differ from the label of the closest prototype.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        return jnp.argmax(self._class_log_posterior(X), axis=1)

    def predict_proba(self, X):
        """Class posteriors :math:`P(c | x)`.

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Columns follow the order of ``classes_``.
        """
        return jnp.exp(self._class_log_posterior(X))

    def predict_with_rejection(self, X, confidence=None):
        """Predict, labelling samples with a low posterior as -1.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        confidence : float, optional
            Minimum class posterior. Defaults to ``rejection_confidence``.

        Returns
        -------
        labels : array of shape (n_samples,)
            Predicted labels, or -1 for rejected samples.
        """
        threshold = confidence if confidence is not None else self.rejection_confidence
        if threshold is None:
            return self.predict(X)
        proba = self.predict_proba(X)
        preds = jnp.argmax(proba, axis=1)
        return jnp.where(jnp.max(proba, axis=1) >= threshold, self._decode_y(preds), -1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        # Hyperparameters of the margin-based parent that a soft model
        # neither uses nor accepts.
        for unused in ('margin', 'beta'):
            hp.pop(unused, None)
        return hp
