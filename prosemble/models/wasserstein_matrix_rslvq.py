"""
Wasserstein Matrix Robust Soft LVQ (WMRSLVQ, WLMRSLVQ).

Robust soft LVQ with distributional prototypes and an adaptive matrix.
Each prototype is a diagonal Gaussian
:math:`\\mathcal{N}(\\mu_k, \\Sigma_k)`, :math:`\\Sigma_k = \\text{diag}(s_k^2)`,
a sample is a Dirac measure, and both are mapped by
:math:`z \\mapsto \\Omega^T z` before they are compared:

.. math::

    W_2^2(x, k) = (x - \\mu_k)^T \\Lambda (x - \\mu_k)
                  + \\operatorname{tr}(\\Lambda \\Sigma_k), \\qquad
    \\Lambda = \\Omega \\Omega^T, \\qquad
    f_k(x) = -\\frac{W_2^2(x, k)}{2\\sigma^2}.

:class:`WMRSLVQ` learns one matrix for all prototypes,
:class:`WLMRSLVQ` one matrix :math:`\\Lambda_k` per prototype.

References
----------
.. [1] Villani, C. (2009). Optimal Transport: Old and New. Springer.
.. [2] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
.. [3] Schneider, P., Biehl, M., & Hammer, B. (2009). Distance learning
       in discriminative vector quantization. Neural Computation.
"""

from prosemble.core.distance import wasserstein2_pushforward_distance_matrix
from prosemble.models.soft_mixin import SoftMetricMixin
from prosemble.models.wasserstein_rslvq import WRSLVQ


class WMRSLVQ(SoftMetricMixin, WRSLVQ):
    """Wasserstein Matrix Robust Soft LVQ.

    :class:`WRSLVQ` with a learned linear map :math:`\\Omega`. The map is
    applied to the sample and to the Gaussian prototype alike, so the
    distance is the squared 2-Wasserstein distance between the mapped
    measures :math:`\\delta_{\\Omega^T x}` and
    :math:`\\mathcal{N}(\\Omega^T \\mu_k, \\Omega^T \\Sigma_k \\Omega)`:

    .. math::

        W_2^2(x, k) = (x - \\mu_k)^T \\Lambda (x - \\mu_k)
                      + \\sum_j \\Lambda_{jj} s_{kj}^2.

    This is the expected adaptive distance from the sample to a point
    drawn from the prototype. The variance of a feature counts in
    proportion to the weight :math:`\\Lambda_{jj}` the metric gives that
    feature, so a prototype may be diffuse along directions the metric
    ignores at no cost. As in :class:`WRSLVQ`, the variance term does not
    depend on the sample and acts as a learned prior weight of the
    prototype.

    The variance term differs from the one of :class:`WGMLVQ`, which adds
    the unweighted total variance :math:`\\sum_j s_{kj}^2`.

    Parameters
    ----------
    latent_dim : int, optional
        Number of columns of :math:`\\Omega`. A value below the number of
        features restricts :math:`\\Lambda` to that rank. Default: the
        number of features.
    sigma : float or 'auto'
        Initial width :math:`\\sigma` of the mixture components.
        ``'auto'`` (default) sets :math:`\\sigma^2` to the mean squared
        distance from a training sample to its closest prototype of the
        same class at initialisation.
    adapt_sigma : bool
        If True (default), learn :math:`\\sigma` by gradient descent on
        the cost. If False, keep it at its initial value.
    rejection_confidence : float, optional
        Minimum class posterior for ``predict_with_rejection``.
        Default: None (no rejection).

    Attributes
    ----------
    omega_ : array of shape (n_features, latent_dim)
        Learned map :math:`\\Omega`.
    sigma_ : float
        Learned width :math:`\\sigma` of the mixture.
    prototype_means_ : array of shape (n_prototypes, n_features)
        Learned prototype means.
    prototype_variances_ : array of shape (n_prototypes, n_features)
        Learned prototype variances.

    See Also
    --------
    WRSLVQ : The model without a matrix, with the remaining parameters.
    WLMRSLVQ : One matrix per prototype.
    MRSLVQ : Matrix robust soft LVQ with point prototypes.
    """

    def __init__(self, latent_dim=None, sigma='auto', adapt_sigma=True,
                 rejection_confidence=None,
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, optimizer='adam',
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            sigma=sigma, adapt_sigma=adapt_sigma,
            rejection_confidence=rejection_confidence,
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
        )
        self._setup_metric(latent_dim)

    def _wasserstein_distances(self, params, X):
        return wasserstein2_pushforward_distance_matrix(
            X, params['prototypes'], params['log_variances'],
            params[self._metric_key])


class WLMRSLVQ(WMRSLVQ):
    """Wasserstein Local Matrix Robust Soft LVQ.

    :class:`WMRSLVQ` with one map :math:`\\Omega_k` per prototype:

    .. math::

        W_2^2(x, k) = (x - \\mu_k)^T \\Lambda_k (x - \\mu_k)
                      + \\operatorname{tr}(\\Lambda_k \\Sigma_k),
        \\qquad \\Lambda_k = \\Omega_k \\Omega_k^T.

    Each prototype measures both its distance to the sample and its own
    spread in its own metric, which gives piecewise quadratic decision
    boundaries. The matrices are not normalised: the size of
    :math:`\\Lambda_k` sets how quickly the score of prototype :math:`k`
    falls off.

    The model has :math:`p \\cdot d \\cdot l` matrix parameters for
    :math:`p` prototypes, :math:`d` features and ``latent_dim``
    :math:`l`, so it needs more data than :class:`WMRSLVQ`.

    Parameters
    ----------
    latent_dim : int, optional
        Number of columns of each :math:`\\Omega_k`. Default: the number
        of features.

    Attributes
    ----------
    omegas_ : array of shape (n_prototypes, n_features, latent_dim)
        Learned maps :math:`\\Omega_k`.
    sigma_ : float
        Learned width :math:`\\sigma` of the mixture.

    See Also
    --------
    WMRSLVQ : One matrix for all prototypes, with the remaining parameters.
    LMRSLVQ : Local matrix robust soft LVQ with point prototypes.
    """

    _local_metric = True
