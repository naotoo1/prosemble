"""
Differentiable Kernel Matrix Robust Soft LVQ (DKMRSLVQ, DKLMRSLVQ).

Robust soft LVQ on the kernel distance of a Gaussian kernel with an
adaptive matrix inside the kernel:

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{(x - w_k)^T \\Lambda (x - w_k)}{2\\gamma_k^2}
    \\right)\\right), \\qquad
    \\Lambda = \\Omega \\Omega^T, \\qquad
    f_k(x) = -\\frac{d_\\kappa^2(x, w_k)}{2\\sigma^2}.

:class:`DKMRSLVQ` learns one matrix for all prototypes,
:class:`DKLMRSLVQ` one matrix :math:`\\Lambda_k` per prototype.

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing, 147,
       83-95.
.. [2] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
.. [3] Schneider, P., Biehl, M., & Hammer, B. (2009). Distance learning
       in discriminative vector quantization. Neural Computation.
"""

import jax.numpy as jnp

from prosemble.core.kernel import kernel_distance_squared_matrix
from prosemble.models.dk_rslvq import DKRSLVQ
from prosemble.models.soft_mixin import SoftMetricMixin


class DKMRSLVQ(SoftMetricMixin, DKRSLVQ):
    """Differentiable Kernel Matrix Robust Soft LVQ.

    :class:`DKRSLVQ` with a learned linear map :math:`\\Omega` inside the
    Gaussian kernel. The kernel
    :math:`\\kappa(x, w) = \\exp(-\\|\\Omega^T (x - w)\\|^2 / 2\\gamma^2)` is
    the Gaussian kernel of the mapped data, so it is positive definite
    for every :math:`\\Omega`, and the kernel distance stays in [0, 2].
    The matrix :math:`\\Lambda = \\Omega \\Omega^T` weights and correlates
    the features: directions that separate the classes are stretched,
    the others shrunk.

    Prototypes, kernel bandwidths :math:`\\gamma_k`, the matrix and the
    mixture width :math:`\\sigma` are learned together on the robust soft
    LVQ cost. The scale of :math:`\\Omega` and the bandwidths are not
    separately identified, since only :math:`\\Lambda / \\gamma_k^2`
    enters the distance.

    Parameters
    ----------
    latent_dim : int, optional
        Number of columns of :math:`\\Omega`. A value below the number of
        features restricts :math:`\\Lambda` to that rank. Default: the
        number of features.
    matrix_lr_ratio : float
        Learning rate of the matrix relative to ``lr``. Default: 0.3.
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
    sigma_init : str or float
        Initialisation of the kernel bandwidths :math:`\\gamma_k`:
        'median' (default), 'mean' or a number.
    sigma_min : float
        Lower bound for the kernel bandwidths. Default: 1e-3.

    Attributes
    ----------
    omega_ : array of shape (n_features, latent_dim)
        Learned map :math:`\\Omega`.
    sigma_ : float
        Learned width :math:`\\sigma` of the mixture.
    sigmas_ : array of shape (n_prototypes,)
        Learned kernel bandwidths :math:`\\gamma_k`.

    See Also
    --------
    DKRSLVQ : The model without a matrix, with the remaining parameters.
    DKLMRSLVQ : One matrix per prototype.
    MRSLVQ : Matrix robust soft LVQ with the squared Euclidean distance.
    """

    def __init__(self, latent_dim=None, matrix_lr_ratio=0.3, sigma='auto',
                 adapt_sigma=True,
                 rejection_confidence=None,
                 sigma_init='median', sigma_min=1e-3,
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, optimizer='adam',
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None,
                 gradient_checkpointing=False, devices=None):
        super().__init__(
            sigma=sigma, adapt_sigma=adapt_sigma,
            rejection_confidence=rejection_confidence,
            sigma_init=sigma_init, sigma_min=sigma_min,
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

    def _kernel_distances(self, params, X):
        bandwidths = jnp.maximum(params['sigmas'], self.sigma_min)
        return kernel_distance_squared_matrix(
            X, params['prototypes'], bandwidths, params[self._metric_key])


class DKLMRSLVQ(DKMRSLVQ):
    """Differentiable Kernel Local Matrix Robust Soft LVQ.

    :class:`DKMRSLVQ` with one map :math:`\\Omega_k` per prototype, so
    that each prototype has its own Gaussian kernel

    .. math::

        \\kappa_k(x, w_k) = \\exp\\left(
            -\\frac{(x - w_k)^T \\Lambda_k (x - w_k)}{2\\gamma_k^2}\\right),
        \\qquad \\Lambda_k = \\Omega_k \\Omega_k^T.

    The decision boundaries between prototypes are then curved in the
    data space as well. The matrices are not normalised: the size of
    :math:`\\Lambda_k` sets how quickly the score of prototype :math:`k`
    falls off, in the same way as its bandwidth.
    Because each prototype has its own matrix, the scores are not the
    log-densities of a normalised mixture (each component would need its
    own normalising term); the class posteriors are those of a
    discriminative model, which is what the cost is the log-likelihood of.

    The model has :math:`p \\cdot d \\cdot l` matrix parameters for
    :math:`p` prototypes, :math:`d` features and ``latent_dim``
    :math:`l`, so it needs more data than :class:`DKMRSLVQ`.

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
    sigmas_ : array of shape (n_prototypes,)
        Learned kernel bandwidths :math:`\\gamma_k`.

    See Also
    --------
    DKMRSLVQ : One matrix for all prototypes, with the remaining parameters.
    LMRSLVQ : Local matrix robust soft LVQ with the squared Euclidean distance.
    """

    _local_metric = True
