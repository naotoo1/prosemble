"""
Differentiating Kernel GRLVQ (DKGRLVQ).

Combines per-feature relevance weighting with per-prototype kernel bandwidth:

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{\\sum_j \\lambda_j (x_j - w_{kj})^2}{2\\sigma_k^2}
    \\right)\\right)

where :math:`\\lambda = \\text{softmax}(\\text{relevances})`.

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.kernel import kernel_distance_squared_relevance


@jit
def _predict_dkgrlvq_jit(X, prototypes, sigmas, relevances, proto_labels,
                          sigma_min):
    """JIT-compiled DKGRLVQ prediction."""
    sigmas = jnp.maximum(sigmas, sigma_min)
    lam = jax.nn.softmax(relevances)
    distances = kernel_distance_squared_relevance(X, prototypes, sigmas, lam)
    return wtac(distances, proto_labels)


class DKGRLVQ(SupervisedPrototypeModel):
    """Differentiating Kernel GRLVQ.

    Combines GRLVQ per-feature relevance weighting with Gaussian kernel
    distance and per-prototype bandwidth adaptation.

    .. math::

        d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
            -\\frac{\\sum_j \\lambda_j (x_j - w_{kj})^2}{2\\sigma_k^2}
        \\right)\\right)

    Parameters
    ----------
    sigma_init : str or float
        Initialization strategy for per-prototype bandwidths.
        'median' (default): per-class median distance from prototype to class members.
        'mean': per-class mean distance.
        float: fixed value for all prototypes.
    sigma_min : float
        Lower bound for sigma to prevent bandwidth collapse. Default: 1e-3.
    beta : float
        Transfer function steepness parameter.
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
    transfer_fn : callable, optional
        Transfer function for loss shaping (default: identity).
    margin : float
        Margin for loss computation.
    callbacks : list, optional
        List of Callback objects.
    use_scan : bool
        If True (default), use jax.lax.scan for training.
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

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters.
    """

    def __init__(self, sigma_init='median', sigma_min=1e-3, beta=10.0,
                 n_prototypes_per_class=1, max_iter=100,
                 lr=0.01, epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None, mixed_precision=None,
                 **kwargs):
        super().__init__(
            n_prototypes_per_class=n_prototypes_per_class,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, distance_fn=distance_fn,
            optimizer=optimizer, transfer_fn=transfer_fn, margin=margin,
            callbacks=callbacks, use_scan=use_scan, batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            prototypes_initializer=prototypes_initializer,
            patience=patience, restore_best=restore_best,
            class_weight=class_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ema_decay=ema_decay, freeze_params=freeze_params,
            lookahead=lookahead, mixed_precision=mixed_precision,
            **kwargs,
        )
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.beta = beta
        self.sigmas_ = None
        self.relevances_ = None

    def _estimate_sigmas(self, X, y, prototypes, proto_labels):
        """Estimate per-prototype bandwidths from data."""
        if isinstance(self.sigma_init, (int, float)):
            return jnp.full(prototypes.shape[0], float(self.sigma_init))

        sigmas = []
        for k in range(prototypes.shape[0]):
            label_k = proto_labels[k]
            class_mask = (y == label_k)
            X_class = X[class_mask]
            dists = jnp.sqrt(jnp.sum((X_class - prototypes[k]) ** 2, axis=1))
            if self.sigma_init == 'median':
                sigma_k = jnp.median(dists)
            else:  # 'mean'
                sigma_k = jnp.mean(dists)
            sigmas.append(jnp.maximum(sigma_k, self.sigma_min))
        return jnp.array(sigmas)

    def _get_resume_params(self, params):
        params['sigmas'] = self.sigmas_
        params['relevances'] = self.relevances_
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        key1, key2 = jax.random.split(key)
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        sigmas = self._estimate_sigmas(X, y, prototypes, proto_labels)
        relevances = jnp.ones(n_features) / n_features
        params = {
            'prototypes': prototypes,
            'relevances': relevances,
            'sigmas': sigmas,
        }
        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        relevances = params['relevances']
        sigmas = jnp.maximum(params['sigmas'], self.sigma_min)
        lam = jax.nn.softmax(relevances)
        distances = kernel_distance_squared_relevance(
            X, prototypes, sigmas, lam
        )
        from prosemble.core.losses import glvq_loss_with_transfer
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _post_update(self, params):
        params['sigmas'] = jnp.maximum(params['sigmas'], self.sigma_min)
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.sigmas_ = params['sigmas']
        self.relevances_ = params['relevances']

    @property
    def relevance_profile(self):
        """Return the learned relevance weights (normalized via softmax)."""
        if self.relevances_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return jax.nn.softmax(self.relevances_)

    @property
    def kernel_bandwidths(self):
        """Return the learned per-prototype bandwidths."""
        if self.sigmas_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.sigmas_

    def _compute_distances_for_rejection(self, X):
        """Relevance kernel distances for reject option."""
        sigmas = jnp.maximum(self.sigmas_, self.sigma_min)
        lam = jax.nn.softmax(self.relevances_)
        return kernel_distance_squared_relevance(X, self.prototypes_, sigmas, lam)

    def predict(self, X):
        """Predict using learned kernel distance with relevance weighting."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_dkgrlvq_jit(
            X, self.prototypes_, self.sigmas_, self.relevances_,
            self.prototype_labels_, self.sigma_min,
        )

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.sigmas_ is not None:
            attrs.append('sigmas_')
        if self.relevances_ is not None:
            attrs.append('relevances_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.sigmas_ is not None:
            arrays['sigmas_'] = np.asarray(self.sigmas_)
        if self.relevances_ is not None:
            arrays['relevances_'] = np.asarray(self.relevances_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'sigmas_' in arrays:
            self.sigmas_ = jnp.asarray(arrays['sigmas_'])
        if 'relevances_' in arrays:
            self.relevances_ = jnp.asarray(arrays['relevances_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        hp['sigma_init'] = self.sigma_init
        hp['sigma_min'] = self.sigma_min
        return hp
