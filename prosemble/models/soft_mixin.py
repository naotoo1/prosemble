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

from prosemble.core.initializers import identity_omega_init
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

    def _add_metric_params(self, params, n_features):
        """Add the parameters of an adaptive metric, if the model has one.

        Called at initialisation before the width is set, so that the
        width is derived from the distance the model trains with.
        """
        return params

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


class SoftMetricMixin:
    """An adaptive linear map inside the distance of a soft model.

    Differences :math:`v` between a sample and a prototype are measured
    as :math:`\\|\\Omega^T v\\|^2 = v^T \\Lambda v` with
    :math:`\\Lambda = \\Omega \\Omega^T`. The map is either shared by
    all prototypes (``params['omega']``, shape (d, l)) or separate for
    each prototype (``params['omegas']``, shape (p, d, l)) when the class
    sets ``_local_metric = True``. It is initialised to the identity, so
    training starts from the model without a metric.

    Place it before the soft model in the bases. The soft model calls
    :meth:`_add_metric_params` when it initialises, and its distance
    reads the map from ``params``.
    """

    _hyperparams = ('latent_dim',)
    _fitted_array_names = ('omega_', 'omegas_')
    _local_metric = False

    def _setup_metric(self, latent_dim):
        if latent_dim is not None and not (
                isinstance(latent_dim, int) and latent_dim > 0):
            raise ValueError(
                f"latent_dim must be a positive integer or None, got {latent_dim!r}")
        self.latent_dim = latent_dim
        setattr(self, self._metric_attr, None)

    @property
    def _metric_key(self):
        return 'omegas' if self._local_metric else 'omega'

    @property
    def _metric_attr(self):
        return self._metric_key + '_'

    def _add_metric_params(self, params, n_features):
        omega = identity_omega_init(n_features, self.latent_dim or n_features)
        if self._local_metric:
            n_prototypes = params['prototypes'].shape[0]
            omega = jnp.tile(omega[None, :, :], (n_prototypes, 1, 1))
        return {**params, self._metric_key: omega}

    @staticmethod
    def _apply_metric(differences, omega):
        """Map differences of shape (n, p, d) to shape (n, p, l)."""
        if omega.ndim == 3:
            return jnp.einsum('npd,pdl->npl', differences, omega)
        return jnp.einsum('npd,dl->npl', differences, omega)

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params[self._metric_key] = getattr(self, self._metric_attr)
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        setattr(self, self._metric_attr, params[self._metric_key])

    def _fitted_params(self):
        return {**super()._fitted_params(),
                self._metric_key: getattr(self, self._metric_attr)}

    @property
    def omega_matrix(self):
        """The learned map :math:`\\Omega`: shape (d, l), or (p, d, l) if local."""
        omega = getattr(self, self._metric_attr)
        if omega is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return omega

    @property
    def lambda_matrix(self):
        """:math:`\\Lambda = \\Omega \\Omega^T`: shape (d, d), or (p, d, d) if local."""
        omega = self.omega_matrix
        if omega.ndim == 3:
            return jnp.einsum('pdl,pel->pde', omega, omega)
        return omega @ omega.T
