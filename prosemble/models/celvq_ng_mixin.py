"""
Cross-Entropy Neural Gas Cooperation Mixin for CELVQ-NG variants.

Provides shared gamma scheduling, NG rank-weighted per-class distance
pooling, and cross-entropy loss computation. Subclasses override
`_compute_distances` to define the metric-specific distance.

This mixin consolidates the identical logic shared by CELVQ_NG,
MCELVQ_NG, LCELVQ_NG, and TCELVQ_NG.
"""

import jax
import jax.numpy as jnp
import numpy as np


class CELVQNGMixin:
    """Mixin that adds Neural Gas cooperation to Cross-Entropy LVQ variants.

    For each class, prototypes are ranked by distance and weighted by
    exp(-rank / gamma). The NG-weighted class distances become logits
    for cross-entropy loss over all classes simultaneously.

    gamma decays from gamma_init -> gamma_final during training.

    Subclasses must override `_compute_distances(params, X)` to return
    a (n_samples, n_prototypes) distance matrix.

    Parameters
    ----------
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
        If True (default), use jax.lax.scan for training (faster, JIT-compiled,
        but runs all max_iter iterations even after convergence).
        If False, use a Python for-loop with true early stopping (no wasted
        compute after convergence, but slower per iteration).
    batch_size : int, optional
        Mini-batch size. If None (default), use full-batch training.
        When set, each epoch iterates over shuffled mini-batches of this size.
    lr_scheduler : str or optax.Schedule, optional
        Learning rate schedule. Supported strings: 'exponential_decay',
        'cosine_decay', 'warmup_cosine_decay', 'warmup_exponential_decay',
        'warmup_constant', 'polynomial', 'linear', 'piecewise_constant',
        'sgdr'. Or pass a custom optax.Schedule. Default: None.
    lr_scheduler_kwargs : dict, optional
        Keyword arguments passed to the learning rate scheduler
        (e.g. ``decay_rate``, ``transition_steps``). Default: None.
    prototypes_initializer : str or callable, optional
        How to initialize prototypes. Supported strings: 'stratified_random'
        (default), 'class_mean', 'class_conditional_mean', 'stratified_noise',
        'random_normal', 'uniform', 'zeros', 'ones', 'fill_value'.
        Or pass a callable ``(X, y, n_per_class, key) -> (protos, labels)``.
    patience : int, optional
        Number of consecutive epochs with no improvement before stopping.
        If None (default), stops after a single non-improving step (epsilon
        check). Requires use_scan=False for true early stopping.
    restore_best : bool
        If True, restore the parameters that achieved the lowest loss
        (or validation loss if validation data is provided). Default: False.
    class_weight : dict or 'balanced', optional
        Weights for each class. Dict maps class label to weight, e.g.
        {0: 1.0, 1: 2.0, 2: 1.5}. 'balanced' auto-computes weights
        inversely proportional to class frequencies. Default: None (uniform).
    gradient_accumulation_steps : int, optional
        Accumulate gradients over this many steps before applying an update.
        Effective batch size = batch_size * gradient_accumulation_steps.
        Default: None (no accumulation).
    ema_decay : float, optional
        Exponential moving average decay for parameters (0 < ema_decay < 1).
        After training, model parameters are replaced with EMA-smoothed values.
        Typical values: 0.999, 0.9999. Default: None (no EMA).
    freeze_params : list of str, optional
        List of parameter group names to freeze (zero gradients).
        E.g. ['backbone'] to freeze the backbone and only train prototypes.
        Default: None (all parameters trainable).
    lookahead : dict, optional
        Enable lookahead optimizer wrapper. Dict with keys:
        - 'sync_period': int (default 6) -- sync every k steps
        - 'slow_step_size': float (default 0.5) -- interpolation factor
        Default: None (no lookahead).
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training. 'float16' or 'bfloat16'.
        Master weights stay in float32; forward/backward pass runs in lower
        precision for ~2x speed and ~half memory on GPU. Float16 uses static
        loss scaling to prevent gradient underflow. Default: None (disabled).
    """

    def __init__(self, gamma_init=None, gamma_final=0.01,
                 gamma_decay=None, n_prototypes_per_class=1,
                 max_iter=100, lr=0.01, epsilon=1e-6, random_seed=42,
                 distance_fn=None, optimizer='adam', transfer_fn=None,
                 margin=0.0, callbacks=None, use_scan=True,
                 batch_size=None, lr_scheduler=None,
                 lr_scheduler_kwargs=None, prototypes_initializer=None,
                 patience=None, restore_best=False, class_weight=None,
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
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.gamma_ = None

        # Freeze gamma from optimizer (not trainable)
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

    def _compute_gamma_init(self):
        """Compute gamma_init from prototype count if not set."""
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)
        gamma_init = self.gamma_init if self.gamma_init is not None else max_per_class / 2.0
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        # Compute decay factor
        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / self.max_iter)

        return gamma_init

    def _init_state(self, X, y, key):
        key1, key2 = jax.random.split(key)
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )

        gamma_init = self._compute_gamma_init()

        params = {
            'prototypes': prototypes,
            'gamma': jnp.array(gamma_init, dtype=jnp.float32),
        }

        # Allow subclasses to add metric params (omega, omegas, etc.)
        params = self._init_metric_params(params, X, prototypes, key2)

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

    def _init_metric_params(self, params, X, prototypes, key):
        """Override to add metric-specific params (omega, omegas, etc.)."""
        return params

    def _compute_distances(self, params, X):
        """Compute distance matrix (n_samples, n_prototypes).

        Must be overridden by subclasses to define metric-specific distance.
        """
        raise NotImplementedError

    def _compute_loss(self, params, X, y, proto_labels):
        gamma = params['gamma']
        n_classes = self.n_classes_

        # 1. Compute distances (n, p)
        distances = self._compute_distances(params, X)

        # 2. NG-weighted per-class distance pooling
        INF = jnp.finfo(distances.dtype).max
        class_dists_list = []

        for c in range(n_classes):
            # Mask: which prototypes belong to class c
            class_mask = (proto_labels == c)  # (p,)

            # Distances to class c prototypes (INF for non-class)
            d_class = jnp.where(class_mask[None, :], distances, INF)  # (n, p)

            # Rank within class c (double argsort)
            order = jnp.argsort(d_class, axis=1)
            ranks = jnp.argsort(order, axis=1).astype(jnp.float32)  # (n, p)

            # NG neighborhood function
            h = jnp.exp(-ranks / (gamma + 1e-10))  # (n, p)
            h = jnp.where(class_mask[None, :], h, 0.0)  # zero non-class

            # Normalize within class
            C = jnp.sum(h, axis=1, keepdims=True)  # (n, 1)
            h_normalized = h / (C + 1e-10)  # (n, p)

            # NG-weighted class distance
            weighted_dist = jnp.sum(h_normalized * distances, axis=1)  # (n,)
            class_dists_list.append(weighted_dist)

        # 3. Stack into (n, n_classes)
        class_dists = jnp.stack(class_dists_list, axis=1)  # (n, n_classes)

        # 4. Cross-entropy loss
        logits = -class_dists  # negate: smaller distance = larger logit
        log_probs = jax.nn.log_softmax(logits, axis=1)
        target_one_hot = jax.nn.one_hot(y, n_classes)
        return -jnp.mean(jnp.sum(target_one_hot * log_probs, axis=1))

    def _post_update(self, params):
        # Decay gamma (neighborhood range) each step
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
