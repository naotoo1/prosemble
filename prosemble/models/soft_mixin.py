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
import numpy as np
import optax

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
    _fitted_array_names = ('sigma_', 'temperature_')

    def _setup_soft(self, sigma, adapt_sigma, rejection_confidence):
        if not (sigma == 'auto' or (isinstance(sigma, (int, float)) and sigma > 0)):
            raise ValueError(f"sigma must be 'auto' or a positive number, got {sigma!r}")
        self.sigma = sigma
        self.adapt_sigma = adapt_sigma
        self.rejection_confidence = rejection_confidence
        self.sigma_ = None
        self.temperature_ = None
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

    def _class_log_posterior(self, X, calibrated=True):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        logits = self._logits(self._fitted_params(), X)
        if calibrated and self.temperature_ is not None:
            logits = logits / self.temperature_
        return class_log_posterior(logits, self.prototype_labels_, self.n_classes_)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(self, *args, **kwargs):
        """Fit the model; a temperature from an earlier :meth:`calibrate` is dropped."""
        self.temperature_ = None
        return super().fit(*args, **kwargs)

    def calibrate(self, X, y, cv=None):
        """Fit a temperature that makes the class posteriors calibrated.

        The width :math:`\\sigma` is learned on the training data, which
        the model fits closely, so the posteriors come out too confident.
        Calibration divides the component scores by a temperature
        :math:`T` chosen to minimise the negative log-likelihood of
        labelled data the model has not been fitted to:

        .. math::

            P_T(c \\mid x) \\propto \\sum_{k: c(k) = c} e^{f_k(x) / T}.

        This is the same as rescaling :math:`\\sigma` by
        :math:`\\sqrt{T}` after training. The temperature applies to
        :meth:`predict_proba` and :meth:`predict_with_rejection`;
        :meth:`predict` keeps the labels of the trained model. (With
        several prototypes per class a temperature can reorder the class
        posteriors of a sample, and on a benchmark of 13 datasets that
        cost accuracy.)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        cv : int, optional
            If None (default), ``X, y`` are held-out data and the
            temperature is fitted to the scores of this model. If an
            integer, ``X, y`` are the training data: ``cv`` copies of the
            model are fitted on stratified folds, and the temperature is
            fitted to their scores on the left-out folds. The fitted
            model itself is kept.

        Returns
        -------
        self
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        y = self._encode_y(y)
        if cv is None:
            logits = np.asarray(self._logits(self._fitted_params(), X))
            labels = np.asarray(y)
        else:
            if not (isinstance(cv, int) and cv >= 2):
                raise ValueError(f"cv must be None or an integer >= 2, got {cv!r}")
            from sklearn.model_selection import StratifiedKFold
            from prosemble.core.model_selection import clone
            parts, label_parts = [], []
            folds = StratifiedKFold(cv, shuffle=True, random_state=self.random_seed)
            y_original = np.asarray(self._decode_y(y))
            for train, held_out in folds.split(np.zeros(len(y_original)), y_original):
                copy = clone(self)
                copy.fit(X[train], y_original[train])
                # Order the component scores like those of this model.
                parts.append(np.asarray(copy._logits(copy._fitted_params(), X[held_out])))
                label_parts.append(np.asarray(copy._encode_y(y_original[held_out])))
                if not np.array_equal(np.asarray(copy.prototype_labels_),
                                      np.asarray(self.prototype_labels_)):
                    raise ValueError(
                        "A fold lacks a class, so its model has different "
                        "prototypes; use fewer folds.")
            logits, labels = np.concatenate(parts), np.concatenate(label_parts)
        self.temperature_ = jnp.asarray(_fit_temperature(
            logits, labels, np.asarray(self.prototype_labels_), self.n_classes_),
            dtype=jnp.float32)
        return self

    def predict(self, X):
        """Predict the class with the largest posterior :math:`P(c | x)`.

        With several prototypes per class this sums over the prototypes
        of a class and can differ from the label of the closest prototype.
        The posterior of the trained model is used, so :meth:`calibrate`
        does not change the predicted labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        return jnp.argmax(self._class_log_posterior(X, calibrated=False), axis=1)

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
        preds = jnp.argmax(self._class_log_posterior(X, calibrated=False), axis=1)
        proba = self.predict_proba(X)
        confidence_of_label = jnp.take_along_axis(proba, preds[:, None], axis=1)[:, 0]
        return jnp.where(confidence_of_label >= threshold, self._decode_y(preds), -1)

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


def _scale_updates(names, factor):
    """Optax transformation that multiplies the updates of the named parameters."""

    def scale(tree):
        if isinstance(tree, dict):
            # A frozen parameter arrives as an empty placeholder, not an array.
            return {k: (v * factor if k in names and hasattr(v, 'dtype') else v)
                    for k, v in tree.items()}
        if isinstance(tree, tuple) and hasattr(tree, '_fields'):
            return type(tree)(*(scale(t) for t in tree))
        return tree

    def init(params):
        return optax.EmptyState()

    def update(updates, state, params=None):
        return scale(updates), state

    return optax.GradientTransformation(init, update)


class SoftMetricMixin:
    """An adaptive linear map inside the distance of a soft model.

    Differences :math:`v` between a sample and a prototype are measured
    as :math:`\\|\\Omega^T v\\|^2 = v^T \\Lambda v` with
    :math:`\\Lambda = \\Omega \\Omega^T`. The map is either shared by
    all prototypes (``params['omega']``, shape (d, l)) or separate for
    each prototype (``params['omegas']``, shape (p, d, l)) when the class
    sets ``_local_metric = True``. It is initialised to the identity, so
    training starts from the model without a metric.

    The map is trained more slowly than the other parameters: its update
    is multiplied by ``matrix_lr_ratio``, which amounts to a learning
    rate of ``matrix_lr_ratio * lr`` for the map. A metric that adapts
    as fast as the prototypes fits the training set before the
    prototypes have settled.

    Place it before the soft model in the bases. The soft model calls
    :meth:`_add_metric_params` when it initialises, and its distance
    reads the map from ``params``.
    """

    _hyperparams = ('latent_dim', 'matrix_lr_ratio')
    _fitted_array_names = ('omega_', 'omegas_')
    _local_metric = False

    def _setup_metric(self, latent_dim, matrix_lr_ratio=0.3):
        if latent_dim is not None and not (
                isinstance(latent_dim, int) and latent_dim > 0):
            raise ValueError(
                f"latent_dim must be a positive integer or None, got {latent_dim!r}")
        if not (isinstance(matrix_lr_ratio, (int, float)) and matrix_lr_ratio > 0):
            raise ValueError(
                f"matrix_lr_ratio must be a positive number, got {matrix_lr_ratio!r}")
        self.latent_dim = latent_dim
        self.matrix_lr_ratio = matrix_lr_ratio
        if matrix_lr_ratio != 1:
            self._optimizer = optax.chain(
                self._optimizer,
                _scale_updates(('omega', 'omegas'), float(matrix_lr_ratio)))
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


def _fit_temperature(logits, labels, prototype_labels, n_classes):
    """Temperature that minimises the negative log-likelihood of ``labels``.

    One-dimensional: a log-spaced grid, then golden-section search in
    the best bracket.
    """
    logits = np.asarray(logits, dtype=np.float64)
    member = prototype_labels[None, :] == np.arange(n_classes)[:, None]   # (classes, prototypes)

    def nll(log_t):
        z = logits / np.exp(log_t)
        z = z - z.max(axis=1, keepdims=True)
        mass = np.exp(z)
        per_class = mass @ member.T                                     # (n, classes)
        own = per_class[np.arange(len(labels)), labels]
        return float(np.mean(np.log(per_class.sum(axis=1)) - np.log(np.maximum(own, 1e-300))))

    grid = np.linspace(np.log(1e-2), np.log(1e2), 81)
    values = [nll(t) for t in grid]
    i = int(np.argmin(values))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(40):
        a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return float(np.exp((lo + hi) / 2.0))
