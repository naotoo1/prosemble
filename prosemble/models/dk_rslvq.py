"""
Differentiable Kernel Robust Soft LVQ (DKRSLVQ).

Robust soft LVQ on the kernel distance of a differentiable Gaussian
kernel with one learnable bandwidth :math:`\\gamma_k` per prototype:

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{\\|x - w_k\\|^2}{2\\gamma_k^2}
    \\right)\\right), \\qquad
    f_k(x) = -\\frac{d_\\kappa^2(x, w_k)}{2\\sigma^2}.

Prototypes stay in the data space; the gradient is taken through the
kernel.

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing, 147,
       83-95.
.. [2] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
"""

import jax.numpy as jnp

from prosemble.core.kernel import kernel_distance_squared_per_proto
from prosemble.models.dk_glvq import DKGLVQ
from prosemble.models.soft_mixin import SoftAssignmentMixin


class DKRSLVQ(SoftAssignmentMixin, DKGLVQ):
    """Differentiable Kernel Robust Soft LVQ.

    The soft counterpart of :class:`DKGLVQ`. Each prototype :math:`w_k`
    has a kernel bandwidth :math:`\\gamma_k`, and the kernel distance
    :math:`d_\\kappa^2(x, w_k) = 2 (1 - \\exp(-\\|x - w_k\\|^2 / 2 \\gamma_k^2))`
    replaces the squared Euclidean distance in the robust soft LVQ cost

    .. math::

        E = -\\sum_i \\log \\frac{\\sum_{k: c(k) = y_i} \\exp(-d_\\kappa^2(x_i, w_k) / 2\\sigma^2)}
                              {\\sum_k \\exp(-d_\\kappa^2(x_i, w_k) / 2\\sigma^2)}.

    For a single kernel this is robust soft LVQ in the feature space of
    the kernel, with Gaussian components centred at the images of the
    prototypes. Prototypes, bandwidths and the width :math:`\\sigma` are
    learned together.

    The kernel distance is bounded by 2, so the scores are bounded by
    :math:`1 / \\sigma^2`: a sample far from every prototype receives
    nearly uniform class posteriors. The scores do not define a
    normalisable density on the data space, and the bandwidths give each
    prototype its own kernel; the likelihood-ratio reading is exact only
    for one shared bandwidth.

    Parameters
    ----------
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
        Initialisation of the kernel bandwidths :math:`\\gamma_k`.
        'median' (default): per-class median distance from the prototype
        to the class members. 'mean': per-class mean distance. float:
        the same value for all prototypes.
    sigma_min : float
        Lower bound of the kernel bandwidths. Default: 1e-3.
    n_prototypes_per_class : int
        Number of prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    epsilon : float
        Convergence threshold on loss change.
    random_seed : int
        Random seed for reproducibility.
    optimizer : str or optax optimizer, optional
        Optimizer name ('adam', 'sgd') or optax GradientTransformation.
    callbacks : list, optional
        List of Callback objects.
    use_scan : bool
        If True, use jax.lax.scan for training.
    batch_size : int, optional
        Mini-batch size. If None, use full-batch training.
    lr_scheduler : str or optax.Schedule, optional
        Learning rate schedule.
    lr_scheduler_kwargs : dict, optional
        Keyword arguments for the learning rate scheduler.
    prototypes_initializer : str or callable, optional
        How to initialize prototypes.
    patience : int, optional
        Epochs with no improvement before stopping.
    restore_best : bool
        If True, restore best parameters after training.
    class_weight : dict or 'balanced', optional
        Weights for each class.
    gradient_accumulation_steps : int, optional
        Accumulate gradients over this many steps.
    ema_decay : float, optional
        Exponential moving average decay for parameters.
    freeze_params : list of str, optional
        Parameter group names to freeze.
    lookahead : dict, optional
        Lookahead optimizer wrapper configuration.
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training.

    Attributes
    ----------
    sigma_ : float
        Learned width :math:`\\sigma` of the mixture.
    sigmas_ : array of shape (n_prototypes,)
        Learned kernel bandwidths :math:`\\gamma_k`.

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing,
           147, 83-95.
    .. [2] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
           Quantization. Neural Computation, 15(7), 1589-1604.

    See Also
    --------
    DKGLVQ : Margin-based counterpart.
    RSLVQ : Robust soft LVQ with the squared Euclidean distance.
    """

    def __init__(self, sigma='auto', adapt_sigma=True,
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
                 mixed_precision=None):
        super().__init__(
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
        )
        self._setup_soft(sigma, adapt_sigma, rejection_confidence)

    def _kernel_distances(self, params, X):
        bandwidths = jnp.maximum(params['sigmas'], self.sigma_min)
        return kernel_distance_squared_per_proto(X, params['prototypes'], bandwidths)

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)
        params = self._add_metric_params(params, X.shape[1])
        params['log_sigma'] = self._initial_log_sigma(
            self._kernel_distances(params, X), y, proto_labels)
        state = state._replace(opt_state=self._optimizer.init(params))
        return state, params, proto_labels

    def _logits(self, params, X):
        return self._scale_by_sigma(
            self._kernel_distances(params, X), params['log_sigma'])

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['log_sigma'] = jnp.log(self.sigma_)
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.sigma_ = jnp.exp(params['log_sigma'])

    def _fitted_params(self):
        return {'prototypes': self.prototypes_, 'sigmas': self.sigmas_,
                'log_sigma': jnp.log(self.sigma_)}
