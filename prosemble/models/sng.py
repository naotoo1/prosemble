"""
Supervised Neural Gas (SNG).

Combines GLVQ's classification loss with Neural Gas neighborhood
cooperation using plain squared Euclidean distance. Unlike SRNG,
SNG does not adapt per-feature relevance weights — it operates
purely in the input space with isotropic distance.

References
----------
.. [1] Hammer, B., Strickert, M., & Villmann, T. (2005). Supervised
       Neural Gas and Extensions. In Proceedings of the Workshop on
       New Challenges in Neural Computation (NC2).
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel


class SNG(SupervisedPrototypeModel):
    """Supervised Neural Gas.

    Combines two key ideas:

    - GLVQ loss: :math:`(d^+ - d^-) / (d^+ + d^-)` for margin-based classification
    - Neural Gas cooperation: all same-class prototypes participate in
      the loss, weighted by rank via :math:`\\exp(-\\text{rank} / \\gamma)`

    Uses plain squared Euclidean distance without any metric adaptation.
    The neighborhood range :math:`\\gamma` decays during training from
    :math:`\\gamma_{\\text{init}}` to :math:`\\gamma_{\\text{final}}`.
    When :math:`\\gamma \\to 0`, SNG recovers standard GLVQ.

    Parameters
    ----------
    beta : float
        Transfer function steepness parameter for sigmoid shaping.
    gamma_init : float, optional
        Initial neighborhood range for NG cooperation.
        Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay factor for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
    lr_ratio : float
        Ratio of wrong-class to correct-class learning rate (ε⁻/ε⁺).
        Default: 0.5.
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
    distance_fn : callable, optional
        Distance function (default: squared Euclidean).
    optimizer : str or optax optimizer, optional
        Optimizer name ('adam', 'sgd') or optax GradientTransformation.
        Default: 'adam'.
    transfer_fn : callable, optional
        Transfer function for loss shaping (default: identity).
    margin : float
        Margin for loss computation.
    callbacks : list, optional
        List of Callback objects.
    use_scan : bool
        If True (default), use jax.lax.scan for training.
    batch_size : int, optional
        Mini-batch size. If None (default), use full-batch training.
    lr_scheduler : str or optax.Schedule, optional
        Learning rate schedule.
    lr_scheduler_kwargs : dict, optional
        Keyword arguments for the learning rate scheduler.
    prototypes_initializer : str or callable, optional
        How to initialize prototypes.
    patience : int, optional
        Number of consecutive epochs with no improvement before stopping.
    restore_best : bool
        If True, restore parameters that achieved lowest loss. Default: False.
    class_weight : dict or 'balanced', optional
        Weights for each class. Default: None (uniform).
    gradient_accumulation_steps : int, optional
        Accumulate gradients over this many steps before applying.
    ema_decay : float, optional
        Exponential moving average decay for parameters.
    freeze_params : list of str, optional
        Parameter group names to freeze.
    lookahead : dict, optional
        Enable lookahead optimizer wrapper.
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training.
    """

    def __init__(self, beta=10.0, gamma_init=None, gamma_final=0.01,
                 gamma_decay=None, lr_ratio=0.5,
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
        self.beta = beta
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.lr_ratio = lr_ratio
        self.gamma_ = None

        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
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
            if self.batch_size is not None:
                steps_per_epoch = (X.shape[0] + self.batch_size - 1) // self.batch_size
            else:
                steps_per_epoch = 1
            total_steps = self.max_iter * steps_per_epoch
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / total_steps)

        params = {
            'prototypes': prototypes,
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
        gamma = params['gamma']

        # Squared Euclidean distance
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        distances = jnp.sum(diff ** 2, axis=2)  # (n, p)

        # Compute ranks within same-class prototypes
        same_class = (y[:, None] == proto_labels[None, :])  # (n, p)
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)  # (n, p)

        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)  # (n, p)

        # Neighborhood function h = exp(-rank / gamma)
        h = jnp.exp(-ranks / (gamma + 1e-10))  # (n, p)
        h = jnp.where(same_class, h, 0.0)

        # Normalize
        C = jnp.sum(h, axis=1, keepdims=True)  # (n, 1)
        h_normalized = h / (C + 1e-10)  # (n, p)

        # Closest different-class prototype distance
        d_diff = jnp.where(~same_class, distances, INF)
        dm = jnp.min(d_diff, axis=1)  # (n,)

        # Separate learning rates
        dm = jax.lax.stop_gradient(dm) + self.lr_ratio * (
            dm - jax.lax.stop_gradient(dm))

        # GLVQ mu
        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)  # (n, p)

        # Transfer function
        from prosemble.core.activations import sigmoid_beta
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)  # (n, p)

        # Rank-weighted sum
        weighted_cost = jnp.sum(h_normalized * cost, axis=1)  # (n,)
        return jnp.mean(weighted_cost)

    def _post_update(self, params):
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
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
        hp['beta'] = self.beta
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        hp['lr_ratio'] = self.lr_ratio
        return hp
