"""
Soft-DTW Robust Soft LVQ (SoftDTWRSLVQ).

Robust soft LVQ for time series. Prototypes are themselves time series,
and the score of a prototype is built on the soft-DTW divergence, which
compares two series under the best monotone alignment of their time
points:

.. math::

    f_k(x) = -\\frac{D_\\gamma(x, w_k)}{2\\sigma^2}.

References
----------
.. [1] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a Differentiable Loss
       Function for Time-Series. Proceedings of the 34th International
       Conference on Machine Learning, PMLR 70, 894-903.
.. [2] Blondel, M., Mensch, A., & Vert, J.-P. (2021). Differentiable
       Divergences Between Time Series. Proceedings of the 24th
       International Conference on Artificial Intelligence and
       Statistics, PMLR 130, 3853-3861.
.. [3] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
.. [4] Jain, B. J., & Schultz, D. (2018). Asymmetric learning vector
       quantization for efficient nearest neighbor classification in
       dynamic time warping spaces. Pattern Recognition, 76, 349-366.
"""

import jax
import jax.numpy as jnp

from prosemble.core.soft_dtw import (
    soft_dtw, soft_dtw_divergence_matrix, soft_dtw_matrix,
)
from prosemble.models.prototype_base import SupervisedPrototypeModel, SupervisedState
from prosemble.models.soft_mixin import SoftAssignmentMixin


class SoftDTWRSLVQ(SoftAssignmentMixin, SupervisedPrototypeModel):
    """Soft-DTW Robust Soft LVQ.

    A labelled mixture over time-series prototypes. The distance is the
    soft-DTW divergence

    .. math::

        D_\\gamma(x, w) = \\operatorname{sdtw}_\\gamma(x, w)
            - \\tfrac12 \\operatorname{sdtw}_\\gamma(x, x)
            - \\tfrac12 \\operatorname{sdtw}_\\gamma(w, w),

    which is non-negative and zero only for equal series. Soft-DTW
    aligns the two series before comparing them, so a prototype matches
    samples whose shape is the same but shifted or stretched in time.
    The prototypes, learned by gradient descent through the alignment,
    are time series of the training length and can be plotted next to
    the data. Training maximises the likelihood ratio of robust soft LVQ.

    The term :math:`\\tfrac12 \\operatorname{sdtw}_\\gamma(x, x)` is the
    same for every prototype, so it cancels in the class posteriors and
    is not computed during training or prediction.

    The cost of one evaluation grows with the product of the series
    lengths, and training keeps the alignment matrices of all
    sample-prototype pairs in memory. For long series or many samples,
    set ``batch_size`` or ``band``.

    Parameters
    ----------
    gamma : float
        Smoothing of soft-DTW. Small values approach DTW; large values
        average over many alignments. The ground cost is the squared
        difference of the time points, so ``gamma`` is on the scale of
        that cost. Default: 0.1.
    band : float, optional
        Sakoe-Chiba band as a fraction of the series length: alignments
        that shift a time point by more than this fraction are excluded.
        Default: None (no band).
    n_channels : int
        Number of channels of a multivariate series. A sample of length
        :math:`T` with :math:`C` channels is a row of :math:`T \\cdot C`
        values, time-major. Default: 1.
    sigma : float or 'auto'
        Initial width :math:`\\sigma` of the mixture components.
        ``'auto'`` (default) sets :math:`\\sigma^2` to the mean
        divergence from a training sample to its closest prototype of
        the same class at initialisation.
    adapt_sigma : bool
        If True (default), learn :math:`\\sigma` by gradient descent on
        the cost. If False, keep it at its initial value.
    rejection_confidence : float, optional
        Minimum class posterior for ``predict_with_rejection``.
        Default: None (no rejection).
    n_prototypes_per_class : int
        Number of prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    batch_size : int, optional
        Mini-batch size. If None, use full-batch training.
    **kwargs
        Other parameters of :class:`SupervisedPrototypeModel`.

    Attributes
    ----------
    prototypes_ : array of shape (n_prototypes, length * n_channels)
        Learned prototype series.
    sigma_ : float
        Learned width :math:`\\sigma` of the mixture.

    References
    ----------
    .. [1] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a Differentiable
           Loss Function for Time-Series. ICML, PMLR 70, 894-903.
    .. [2] Blondel, M., Mensch, A., & Vert, J.-P. (2021). Differentiable
           Divergences Between Time Series. AISTATS, PMLR 130, 3853-3861.

    See Also
    --------
    RSLVQ : Robust soft LVQ with the squared Euclidean distance.
    """

    _hyperparams = ('gamma', 'band', 'n_channels')

    def __init__(self, gamma=0.1, band=None, n_channels=1, sigma='auto',
                 adapt_sigma=True, rejection_confidence=None,
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, optimizer='adam',
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None, mixed_precision=None,
                 gradient_checkpointing=False, devices=None):
        super().__init__(
            n_prototypes_per_class=n_prototypes_per_class,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, optimizer=optimizer,
            callbacks=callbacks, use_scan=use_scan, batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            prototypes_initializer=prototypes_initializer,
            patience=patience, restore_best=restore_best,
            class_weight=class_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ema_decay=ema_decay, freeze_params=freeze_params,
            lookahead=lookahead, mixed_precision=mixed_precision,
            gradient_checkpointing=gradient_checkpointing, devices=devices,
        )
        if not (isinstance(gamma, (int, float)) and gamma > 0):
            raise ValueError(f"gamma must be a positive number, got {gamma!r}")
        if band is not None and not (isinstance(band, (int, float)) and 0 < band <= 1):
            raise ValueError(f"band must be None or a number in (0, 1], got {band!r}")
        if not (isinstance(n_channels, int) and n_channels > 0):
            raise ValueError(f"n_channels must be a positive integer, got {n_channels!r}")
        self.gamma = gamma
        self.band = band
        self.n_channels = n_channels
        self._setup_soft(sigma, adapt_sigma, rejection_confidence)

    def _series(self, X):
        """Rows of ``X`` as series of shape (n, length, channels)."""
        if X.shape[1] % self.n_channels:
            raise ValueError(
                f"{X.shape[1]} values per sample is not a multiple of "
                f"n_channels={self.n_channels}")
        return X.reshape(X.shape[0], -1, self.n_channels)

    def _init_state(self, X, y, key):
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key)
        # A model with a metric over the channels adds its map here, so
        # that the initial width is derived from the distance it trains with.
        params = self._add_metric_params({'prototypes': prototypes}, self.n_channels)
        params['log_sigma'] = self._initial_log_sigma(
            jnp.maximum(self._divergence_matrix(params, X), 0.0), y, proto_labels)
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=self._optimizer.init(params),
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _logits(self, params, X):
        series = self._series(X)
        prototypes = self._series(params['prototypes'])
        cross = soft_dtw_matrix(series, prototypes, self.gamma, self.band)
        own = jax.vmap(lambda w: soft_dtw(w, w, self.gamma, self.band))(prototypes)
        # D(x, w) up to the term 0.5 sdtw(x, x), which is the same for
        # every prototype and cancels in the posteriors.
        return self._scale_by_sigma(cross - 0.5 * own[None, :], params['log_sigma'])

    def divergences(self, X):
        """Soft-DTW divergence from each sample to each prototype.

        Parameters
        ----------
        X : array-like of shape (n_samples, length * n_channels)

        Returns
        -------
        D : array of shape (n_samples, n_prototypes)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return self._divergence_matrix(self._fitted_params(), X)

    def _divergence_matrix(self, params, X):
        """Soft-DTW divergence between the rows of ``X`` and the prototypes."""
        return soft_dtw_divergence_matrix(
            self._series(X), self._series(params['prototypes']), self.gamma, self.band)

    def _get_resume_params(self, params):
        params['log_sigma'] = jnp.log(self.sigma_)
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.sigma_ = jnp.exp(params['log_sigma'])

    def _fitted_params(self):
        return {'prototypes': self.prototypes_, 'log_sigma': jnp.log(self.sigma_)}

    def _compute_distances_for_rejection(self, X):
        return self.divergences(X)
