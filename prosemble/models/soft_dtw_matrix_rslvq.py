"""
Soft-DTW Matrix Robust Soft LVQ (SoftDTWMRSLVQ).

SoftDTWRSLVQ for multivariate time series with a metric over the
channels that is learned together with the prototypes. Two time points
are compared by

.. math::

    c(x_t, w_s) = \\|\\Omega^T (x_t - w_s)\\|^2
                = (x_t - w_s)^T \\Lambda (x_t - w_s),
    \\qquad \\Lambda = \\Omega \\Omega^T,

inside the soft-DTW alignment. This equals the squared Euclidean cost
between the projected series :math:`\\Omega^T x_t` and
:math:`\\Omega^T w_s`, so the divergence is the soft-DTW divergence of
the projected series. The diagonal of :math:`\\Lambda` is the relevance
of each channel; off-diagonal entries show channels that act together.

References
----------
.. [1] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a Differentiable Loss
       Function for Time-Series. Proceedings of the 34th International
       Conference on Machine Learning, PMLR 70, 894-903.
.. [2] Blondel, M., Mensch, A., & Vert, J.-P. (2021). Differentiable
       Divergences Between Time Series. Proceedings of the 24th
       International Conference on Artificial Intelligence and
       Statistics, PMLR 130, 3853-3861.
.. [3] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive Relevance
       Matrices in Learning Vector Quantization. Neural Computation,
       21(12), 3532-3561.
"""

import jax
import jax.numpy as jnp

from prosemble.core.soft_dtw import soft_dtw, soft_dtw_divergence_matrix, soft_dtw_matrix
from prosemble.models.soft_dtw_rslvq import SoftDTWRSLVQ
from prosemble.models.soft_mixin import SoftMetricMixin


class SoftDTWMRSLVQ(SoftMetricMixin, SoftDTWRSLVQ):
    """Soft-DTW Matrix Robust Soft LVQ.

    :class:`SoftDTWRSLVQ` with a learned metric over the channels of a
    multivariate series. Each time point of a sample and of a prototype
    is mapped by :math:`\\Omega^T` before the alignment, so the ground
    cost of soft-DTW is :math:`(x_t - w_s)^T \\Lambda (x_t - w_s)` with
    :math:`\\Lambda = \\Omega \\Omega^T`. The diagonal of ``lambda_matrix``
    is the relevance of each channel.

    The scale of :math:`\\Omega` would also scale the ground cost and so
    change the meaning of ``gamma``. The map is therefore normalised to
    :math:`\\operatorname{tr} \\Lambda = C`, the number of channels: at
    initialisation :math:`\\Lambda = I` and the model is SoftDTWRSLVQ,
    and training only redistributes weight between the channels.

    Parameters
    ----------
    latent_dim : int, optional
        Number of columns of :math:`\\Omega`. A value below the number of
        channels restricts :math:`\\Lambda` to that rank. Default: the
        number of channels.
    matrix_lr_ratio : float
        Learning rate of the matrix relative to ``lr``. Default: 0.3.
    gamma : float
        Smoothing of soft-DTW. Default: 0.1.
    band : float, optional
        Sakoe-Chiba band as a fraction of the series length.
        Default: None (no band).
    n_channels : int
        Number of channels. A sample of length :math:`T` with :math:`C`
        channels is a row of :math:`T \\cdot C` values, time-major.
        Default: 1.
    sigma : float or 'auto'
        Initial width :math:`\\sigma` of the mixture components.
        ``'auto'`` (default) sets it from the training data.
    adapt_sigma : bool
        If True (default), learn :math:`\\sigma`.
    rejection_confidence : float, optional
        Minimum class posterior for ``predict_with_rejection``.
        Default: None (no rejection).

    Attributes
    ----------
    prototypes_ : array of shape (n_prototypes, length * n_channels)
        Learned prototype series.
    omega_ : array of shape (n_channels, latent_dim)
        Learned map :math:`\\Omega`, normalised to
        :math:`\\operatorname{tr} \\Lambda = C`.
    sigma_ : float
        Learned width :math:`\\sigma` of the mixture.

    See Also
    --------
    SoftDTWRSLVQ : The model without a metric over the channels.
    """

    def __init__(self, latent_dim=None, matrix_lr_ratio=0.3, gamma=0.1, band=None,
                 n_channels=1, sigma='auto', adapt_sigma=True,
                 rejection_confidence=None, n_prototypes_per_class=1,
                 max_iter=100, lr=0.01, epsilon=1e-6, random_seed=42,
                 optimizer='adam', callbacks=None, use_scan=True,
                 batch_size=None, lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None, mixed_precision=None,
                 gradient_checkpointing=False, devices=None):
        super().__init__(
            gamma=gamma, band=band, n_channels=n_channels, sigma=sigma,
            adapt_sigma=adapt_sigma, rejection_confidence=rejection_confidence,
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
        self._setup_metric(latent_dim, matrix_lr_ratio)

    def _normalised_map(self, omega):
        """Scale ``omega`` (or each of a stack) to :math:`\\operatorname{tr} \\Lambda = C`."""
        squared = jnp.sum(omega ** 2, axis=(-2, -1), keepdims=True)
        return omega * jnp.sqrt(self.n_channels / jnp.maximum(squared, 1e-12))

    def _projected(self, params, X):
        """Samples and prototypes as series mapped by the normalised :math:`\\Omega^T`."""
        omega = self._normalised_map(params['omega'])
        series = self._series(X) @ omega
        prototypes = self._series(params['prototypes']) @ omega
        return series, prototypes

    def _logits(self, params, X):
        series, prototypes = self._projected(params, X)
        cross = soft_dtw_matrix(series, prototypes, self.gamma, self.band)
        own = jax.vmap(lambda w: soft_dtw(w, w, self.gamma, self.band))(prototypes)
        # As in SoftDTWRSLVQ, 0.5 sdtw(x, x) is the same for every
        # prototype under a shared map and cancels in the posteriors.
        return self._scale_by_sigma(cross - 0.5 * own[None, :], params['log_sigma'])

    def _divergence_matrix(self, params, X):
        series, prototypes = self._projected(params, X)
        return soft_dtw_divergence_matrix(series, prototypes, self.gamma, self.band)

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        setattr(self, self._metric_attr, self._normalised_map(params[self._metric_key]))

    @property
    def channel_relevances(self):
        """Relevance of each channel: the diagonal of :math:`\\Lambda` divided by C.

        Non-negative and summing to one; shape (n_channels,).
        """
        lam = self.lambda_matrix
        return jnp.diagonal(lam, axis1=-2, axis2=-1) / self.n_channels
