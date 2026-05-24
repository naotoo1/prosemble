"""
Differentiating Kernel GLVQ with Neural Gas cooperation (DKGLVQ-NG).

Combines DKGLVQ's Gaussian kernel distance and per-prototype bandwidth
adaptation with Neural Gas neighborhood cooperation. All same-class
prototypes participate in the loss, weighted by their distance rank.

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{\\|x - w_k\\|^2}{2\\sigma_k^2}
    \\right)\\right)

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.kernel import kernel_distance_squared_per_proto


@jit
def _predict_dkglvq_ng_jit(X, prototypes, sigmas, proto_labels, sigma_min):
    """JIT-compiled prediction with per-prototype kernel distance."""
    sigmas = jnp.maximum(sigmas, sigma_min)
    distances = kernel_distance_squared_per_proto(X, prototypes, sigmas)
    return wtac(distances, proto_labels)


class DKGLVQ_NG(SupervisedPrototypeModel):
    """Differentiating Kernel GLVQ with Neural Gas cooperation.

    Combines Gaussian kernel distance with per-prototype bandwidth
    adaptation and Neural Gas rank-weighted loss. All same-class
    prototypes participate, weighted by :math:`\\exp(-\\text{rank} / \\gamma)`.

    .. math::

        d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
            -\\frac{\\|x - w_k\\|^2}{2\\sigma_k^2}
        \\right)\\right)

    When :math:`\\gamma \\to 0`, DKGLVQ-NG recovers standard DKGLVQ.

    Parameters
    ----------
    sigma_init : str or float
        Initialization strategy for per-prototype bandwidths.
        'median' (default): per-class median distance from prototype
        to class members.
        'mean': per-class mean distance.
        float: fixed value for all prototypes.
    sigma_min : float
        Lower bound for sigma to prevent bandwidth collapse. Default: 1e-3.
    beta : float
        Transfer function steepness. Default: 10.0.
    gamma_init : float, optional
        Initial neighborhood range. Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
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

    Attributes
    ----------
    sigmas_ : array of shape (n_prototypes,)
        Learned per-prototype kernel bandwidths.
    gamma_ : float
        Final gamma value after training.

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.
    .. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
           Neural Gas with General Similarity Measure. Neural Processing
           Letters.

    See Also
    --------
    DKGLVQ : Base variant without NG cooperation.
    SRNG : NG variant with Euclidean relevance-weighted distance.
    """

    def __init__(self, sigma_init='median', sigma_min=1e-3, beta=10.0,
                 gamma_init=None, gamma_final=0.01, gamma_decay=None,
                 n_prototypes_per_class=1, max_iter=100,
                 lr=0.01, epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None, margin=0.0,
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
        )
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.beta = beta
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.sigmas_ = None
        self.gamma_ = None

        # Ensure gamma is frozen from optimizer (not trainable)
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _estimate_sigmas(self, X, y, prototypes, proto_labels):
        """Estimate per-prototype bandwidths from same-class data."""
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
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        key1, key2 = jax.random.split(key)
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        sigmas = self._estimate_sigmas(X, y, prototypes, proto_labels)

        # Gamma initialization (same pattern as SRNG)
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)
        gamma_init = self.gamma_init if self.gamma_init is not None else max_per_class / 2.0
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / self.max_iter)

        params = {
            'prototypes': prototypes,
            'sigmas': sigmas,
            'gamma': jnp.array(gamma_init, dtype=jnp.float32),
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
        sigmas = jnp.maximum(params['sigmas'], self.sigma_min)
        gamma = params['gamma']

        # Kernel distances: (n, p)
        distances = kernel_distance_squared_per_proto(X, prototypes, sigmas)

        # Class-aware ranking (only same-class prototypes)
        same_class = (y[:, None] == proto_labels[None, :])
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)

        # Double argsort for ranks
        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)

        # h = exp(-rank / gamma), zero for wrong-class
        h = jnp.exp(-ranks / (gamma + 1e-10))
        h = jnp.where(same_class, h, 0.0)

        # Normalize over same-class prototypes
        C = jnp.sum(h, axis=1, keepdims=True)
        h_normalized = h / (C + 1e-10)

        # Closest different-class prototype distance
        d_diff = jnp.where(~same_class, distances, INF)
        dm = jnp.min(d_diff, axis=1)

        # GLVQ mu for each (sample, prototype) pair
        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)

        # Apply transfer function
        from prosemble.core.activations import sigmoid_beta
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)

        # Rank-weighted sum over same-class prototypes
        weighted_cost = jnp.sum(h_normalized * cost, axis=1)
        return jnp.mean(weighted_cost)

    def _post_update(self, params):
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {
            **params,
            'sigmas': jnp.maximum(params['sigmas'], self.sigma_min),
            'gamma': new_gamma,
        }

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.sigmas_ = params['sigmas']
        self.gamma_ = float(params['gamma'])

    def _compute_distances_for_rejection(self, X):
        """Kernel distances for reject option."""
        sigmas = jnp.maximum(self.sigmas_, self.sigma_min)
        return kernel_distance_squared_per_proto(X, self.prototypes_, sigmas)

    def predict(self, X):
        """Predict using learned kernel distance."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_dkglvq_ng_jit(
            X, self.prototypes_, self.sigmas_,
            self.prototype_labels_, self.sigma_min,
        )

    @property
    def kernel_bandwidths(self):
        """Return the learned per-prototype bandwidths."""
        if self.sigmas_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.sigmas_

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.sigmas_ is not None:
            attrs.append('sigmas_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.sigmas_ is not None:
            arrays['sigmas_'] = np.asarray(self.sigmas_)
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'sigmas_' in arrays:
            self.sigmas_ = jnp.asarray(arrays['sigmas_'])
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        hp['sigma_init'] = self.sigma_init
        hp['sigma_min'] = self.sigma_min
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        return hp
