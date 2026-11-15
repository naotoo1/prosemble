"""
Wasserstein Robust Soft LVQ (WRSLVQ).

Robust soft LVQ with distributional prototypes. Each prototype is a
diagonal Gaussian :math:`\\mathcal{N}(\\mu_k, \\text{diag}(s_k^2))`, a
sample is a Dirac measure, and the score is built on the squared
2-Wasserstein distance

.. math::

    W_2^2(x, k) = \\|x - \\mu_k\\|^2 + \\sum_j s_{kj}^2, \\qquad
    f_k(x) = -\\frac{W_2^2(x, k)}{2\\sigma^2}.

References
----------
.. [1] Villani, C. (2009). Optimal Transport: Old and New. Springer.
.. [2] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
"""

import jax.numpy as jnp

from prosemble.core.distance import wasserstein2_distance_matrix
from prosemble.models.soft_mixin import SoftAssignmentMixin
from prosemble.models.wasserstein_glvq import WGLVQ


class WRSLVQ(SoftAssignmentMixin, WGLVQ):
    """Wasserstein Robust Soft LVQ.

    The soft counterpart of :class:`WGLVQ`. Prototypes are diagonal
    Gaussians and the squared 2-Wasserstein distance replaces the squared
    Euclidean distance in the robust soft LVQ cost.

    Because :math:`W_2^2(x, k) = \\|x - \\mu_k\\|^2 + \\operatorname{tr}\\Sigma_k`,
    the score separates into a Gaussian term around the mean and a term
    that does not depend on the sample:

    .. math::

        e^{f_k(x)} =
        \\underbrace{\\exp\\left(
            -\\frac{\\operatorname{tr}\\Sigma_k}{2\\sigma^2}
        \\right)}_{\\propto\\, P(k)}
        \\exp\\left(-\\frac{\\|x - \\mu_k\\|^2}{2\\sigma^2}\\right).

    The model is therefore robust soft LVQ with learned prior weights:
    a prototype with a large total variance is a diffuse component with a
    small prior, one with a small total variance a confident component
    with a large prior. Only the total variance of a prototype enters the
    cost.

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
    prototype_means_ : array of shape (n_prototypes, n_features)
        Means of the Gaussian prototypes.
    prototype_variances_ : array of shape (n_prototypes, n_features)
        Variances of the Gaussian prototypes.

    References
    ----------
    .. [1] Villani, C. (2009). Optimal Transport: Old and New. Springer.
    .. [2] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
           Quantization. Neural Computation, 15(7), 1589-1604.

    See Also
    --------
    WGLVQ : Margin-based counterpart.
    """

    def __init__(self, sigma='auto', adapt_sigma=True,
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

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)
        distances = wasserstein2_distance_matrix(
            X, params['prototypes'], params['log_variances'])
        params['log_sigma'] = self._initial_log_sigma(distances, y, proto_labels)
        state = state._replace(opt_state=self._optimizer.init(params))
        return state, params, proto_labels

    def _logits(self, params, X):
        distances = wasserstein2_distance_matrix(
            X, params['prototypes'], params['log_variances'])
        return self._scale_by_sigma(distances, params['log_sigma'])

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['log_variances'] = self._log_variances_
        params['log_sigma'] = jnp.log(self.sigma_)
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.sigma_ = jnp.exp(params['log_sigma'])

    def _fitted_params(self):
        return {'prototypes': self.prototypes_,
                'log_variances': self._log_variances_,
                'log_sigma': jnp.log(self.sigma_)}
