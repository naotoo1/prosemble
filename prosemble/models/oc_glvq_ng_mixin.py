"""
Neural Gas Cooperation Mixin for OC-GLVQ variants.

Provides shared gamma scheduling, NG ranking, and rank-weighted
OC-GLVQ loss computation. Subclasses override `_compute_distances`
to define the metric-specific distance.
"""

import jax.numpy as jnp
import numpy as np

from prosemble.core.activations import sigmoid_beta


class NGCooperationMixin:
    """Mixin that adds Neural Gas neighborhood cooperation to OC-GLVQ variants.

    Instead of only the nearest prototype contributing to the loss, ALL
    prototypes participate weighted by their distance rank:

        h_k = exp(-rank_k / γ)
        E = mean(Σ_k h̃_k · f(μ_k))

    γ decays from gamma_init → gamma_final during training.

    Subclasses must override `_compute_distances(params, X)` to return
    a (n_samples, n_prototypes) distance matrix.

    Parameters
    ----------
    gamma_init : float, optional
        Initial neighborhood range. Default: n_prototypes / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
    """

    def __init__(self, gamma_init=None, gamma_final=0.01,
                 gamma_decay=None, n_prototypes=3, target_label=None,
                 beta=10.0, max_iter=100, lr=0.01, epsilon=1e-6,
                 random_seed=42, distance_fn=None, optimizer='adam',
                 transfer_fn=None, margin=0.0, callbacks=None,
                 use_scan=True, batch_size=None, lr_scheduler=None,
                 lr_scheduler_kwargs=None, prototypes_initializer=None,
                 patience=None, restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None, **kwargs):
        super().__init__(
            n_prototypes=n_prototypes, target_label=target_label,
            beta=beta, max_iter=max_iter, lr=lr, epsilon=epsilon,
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
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.gamma_ = None

        # Freeze gamma from optimizer
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)

        gamma_init = self.gamma_init if self.gamma_init is not None else self.n_prototypes / 2.0
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / self.max_iter)

        params['gamma'] = jnp.array(gamma_init, dtype=jnp.float32)
        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=params['prototypes'],
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_distances(self, params, X):
        """Compute distance matrix (n_samples, n_prototypes).

        Must be overridden by subclasses to define metric-specific distance.
        """
        raise NotImplementedError

    def _compute_loss(self, params, X, y, proto_labels):
        thetas = params['thetas']
        gamma = params['gamma']

        distances = self._compute_distances(params, X)

        # NG ranking: double argsort
        order = jnp.argsort(distances, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)

        # Neighborhood function
        h = jnp.exp(-ranks / (gamma + 1e-10))
        h_norm = h / (jnp.sum(h, axis=1, keepdims=True) + 1e-10)

        # Per-prototype mu for all prototypes
        s = jnp.where(y == self._target_label, 1.0, -1.0)
        mu = s[:, None] * (distances - thetas[None, :]) / (
            distances + thetas[None, :] + 1e-10
        )

        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)

        return jnp.mean(jnp.sum(h_norm * cost, axis=1))

    def _post_update(self, params):
        params = super()._post_update(params)
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.gamma_ = float(params['gamma'])

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        return hp
