"""
Neural Gas Cooperation Mixin for OC-RSLVQ variants.

Combines OC-RSLVQ's Gaussian soft-assignment with Neural Gas rank-based
neighborhood cooperation. Unlike the OC-GLVQ-NG mixin (which replaces
hard nearest-prototype with NG ranking), this mixin modulates the
existing Gaussian responsibilities with NG neighborhood weights:

.. math::

    p(k|x) = \\frac{\\exp(-d_k / 2\\sigma^2)}{\\sum_j \\exp(-d_j / 2\\sigma^2)} \\quad [\\text{Gaussian}]

.. math::

    h_k = \\exp(-\\text{rank}_k / \\gamma) \\quad [\\text{NG neighborhood}]

.. math::

    w_k = \\frac{p(k|x) \\cdot h_k}{\\sum_j p(j|x) \\cdot h_j} \\quad [\\text{combined}]

When :math:`\\gamma \\to \\infty`, :math:`h_k \\approx \\text{const}` for all :math:`k` -- recovers OC-RSLVQ (pure Gaussian).
When :math:`\\gamma \\to 0`, only the nearest prototype has :math:`h_k > 0` -- sharpened assignment.

Subclasses override ``_compute_distances(params, X)`` to define the
metric-specific distance (Euclidean, global :math:`\\Omega`, local :math:`\\Omega_k`).
"""

import jax.numpy as jnp
import numpy as np

from prosemble.core.activations import sigmoid_beta


class OCRSLVQNGMixin:
    """Mixin that adds Neural Gas cooperation to OC-RSLVQ variants.

    Provides gamma scheduling, combined Gaussian+NG weighting, and
    the corresponding loss computation.

    Subclasses must override `_compute_distances(params, X)` to return
    a (n_samples, n_prototypes) distance matrix.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture for prototype weighting.
    gamma_init : float, optional
        Initial neighborhood range. Default: n_prototypes / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.
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

    def __init__(self, sigma=1.0, gamma_init=None, gamma_final=0.01,
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
        self.sigma = sigma
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
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual')
            else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)

        gamma_init = (self.gamma_init if self.gamma_init is not None
                      else self.n_prototypes / 2.0)
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (
                self.gamma_final / gamma_init
            ) ** (1.0 / self.max_iter)

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

        # 1. Gaussian mixture probabilities
        log_probs = -distances / (2.0 * self.sigma ** 2)
        log_norm = jnp.max(log_probs, axis=1, keepdims=True)
        gauss = jnp.exp(log_probs - log_norm)
        gauss = gauss / (jnp.sum(gauss, axis=1, keepdims=True) + 1e-10)

        # 2. NG rank-based neighborhood
        order = jnp.argsort(distances, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)
        h = jnp.exp(-ranks / (gamma + 1e-10))
        h_norm = h / (jnp.sum(h, axis=1, keepdims=True) + 1e-10)

        # 3. Combined weights (renormalized)
        combined = gauss * h_norm
        combined = combined / (jnp.sum(combined, axis=1, keepdims=True)
                               + 1e-10)

        # 4. Per-prototype OC mu
        s = jnp.where(y == self._target_label, 1.0, -1.0)
        mu = s[:, None] * (distances - thetas[None, :]) / (
            distances + thetas[None, :] + 1e-10
        )

        # 5. Weighted sigmoid loss
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)
        return jnp.mean(jnp.sum(combined * cost, axis=1))

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

    def decision_function(self, X):
        """Compute target-likeness scores using combined Gaussian+NG weights.

        Uses final (converged) gamma for NG modulation at inference time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : array of shape (n_samples,)
            Scores near 1.0 indicate target class, near 0.0 indicate outlier.
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        import jax

        distances = self._inference_distances(X)

        # Gaussian weights
        log_probs = -distances / (2.0 * self.sigma ** 2)
        log_norm = jnp.max(log_probs, axis=1, keepdims=True)
        gauss = jnp.exp(log_probs - log_norm)
        gauss = gauss / (jnp.sum(gauss, axis=1, keepdims=True) + 1e-10)

        # NG rank weights (using final gamma)
        order = jnp.argsort(distances, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)
        h = jnp.exp(-ranks / (self.gamma_ + 1e-10))
        h_norm = h / (jnp.sum(h, axis=1, keepdims=True) + 1e-10)

        # Combined
        combined = gauss * h_norm
        combined = combined / (jnp.sum(combined, axis=1, keepdims=True)
                               + 1e-10)

        # Per-prototype mu (from target perspective)
        mu = (distances - self.thetas_[None, :]) / (
            distances + self.thetas_[None, :] + 1e-10
        )
        weighted_mu = jnp.sum(combined * mu, axis=1)
        return 1.0 - jax.nn.sigmoid(self.beta * weighted_mu)

    def _inference_distances(self, X):
        """Compute distances for inference. Override for metric variants."""
        return self.distance_fn(X, self.prototypes_)

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
        hp['sigma'] = self.sigma
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        return hp
