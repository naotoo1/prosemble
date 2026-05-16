"""Base classes for supervised and unsupervised prototype-based models.

SupervisedPrototypeModel: Base for GLVQ, GMLVQ, LVQ, CBC, etc.
UnsupervisedPrototypeModel: Base for NeuralGas, GrowingNeuralGas, KohonenSOM.

Both follow prosemble's established patterns:
- NamedTuple state for JIT compatibility
- Dual training paths (lax.scan vs Python loop with callbacks)
- save/load via NPZ serialization
- Callback system for visualization/monitoring
"""

import json
from abc import ABC, abstractmethod
from typing import NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, lax
from functools import partial

from prosemble.models.base import NotFittedError
from prosemble.core.quantization import MetadataCollectorMixin, QuantizationMixin
from prosemble.core.serialization import SerializationMixin
from prosemble.core.distance import squared_euclidean_distance_matrix
from prosemble.core.competitions import wtac
from prosemble.core.pooling import stratified_min_pooling
from prosemble.core.initializers import (
    stratified_selection_init,
    stratified_mean_init,
    stratified_noise_init,
    class_conditional_mean_init,
    random_normal_init,
    uniform_init,
    zeros_init,
    ones_init,
    fill_value_init,
    literal_init,
)


# --- Prototype initializer registry ---

_PROTOTYPE_INITIALIZER_REGISTRY = {
    # Class-aware initializers (use training data + labels)
    'stratified_random': lambda X, y, n, key, **kw: stratified_selection_init(X, y, n, key),
    'class_mean': lambda X, y, n, key, **kw: stratified_mean_init(X, y),
    'class_conditional_mean': lambda X, y, n, key, **kw: class_conditional_mean_init(X, y, n),
    'stratified_noise': lambda X, y, n, key, **kw: stratified_noise_init(
        X, y, n, key, noise_std=kw.get('noise_std', 0.1)
    ),
    # Dimension-aware initializers (don't use data/labels)
    'random_normal': lambda X, y, n, key, **kw: (
        random_normal_init(
            _total_prototypes(y, n), X.shape[1], key,
            mean=kw.get('mean', 0.0), std=kw.get('std', 1.0),
        ),
        _generate_proto_labels(y, n),
    ),
    'uniform': lambda X, y, n, key, **kw: (
        uniform_init(
            _total_prototypes(y, n), X.shape[1], key,
            low=kw.get('low', 0.0), high=kw.get('high', 1.0),
        ),
        _generate_proto_labels(y, n),
    ),
    'zeros': lambda X, y, n, key, **kw: (
        zeros_init(_total_prototypes(y, n), X.shape[1]),
        _generate_proto_labels(y, n),
    ),
    'ones': lambda X, y, n, key, **kw: (
        ones_init(_total_prototypes(y, n), X.shape[1]),
        _generate_proto_labels(y, n),
    ),
    'fill_value': lambda X, y, n, key, **kw: (
        fill_value_init(_total_prototypes(y, n), X.shape[1],
                        value=kw.get('value', 0.0)),
        _generate_proto_labels(y, n),
    ),
}


def _total_prototypes(y, n_per_class):
    """Compute total number of prototypes from labels and per-class count."""
    classes = jnp.unique(y)
    if isinstance(n_per_class, dict):
        return sum(n_per_class[int(c)] for c in classes)
    elif isinstance(n_per_class, (list, tuple)):
        return sum(n_per_class[int(c)] for c in classes)
    return len(classes) * n_per_class


def _generate_proto_labels(y, n_per_class):
    """Generate prototype labels for dimension-aware initializers."""
    classes = jnp.unique(y)
    all_labels = []
    for c in classes:
        c_int = int(c)
        if isinstance(n_per_class, dict):
            n = n_per_class[c_int]
        elif isinstance(n_per_class, (list, tuple)):
            n = n_per_class[c_int]
        else:
            n = n_per_class
        all_labels.append(jnp.full(n, c, dtype=y.dtype))
    return jnp.concatenate(all_labels, axis=0)


# --- JIT-compiled inference functions ---

@partial(jit, static_argnums=(3,))
def _predict_supervised_jit(X, prototypes, proto_labels, distance_fn):
    """JIT-compiled supervised prediction pipeline."""
    distances = distance_fn(X, prototypes)
    return wtac(distances, proto_labels)


@partial(jit, static_argnums=(3, 4))
def _predict_proba_supervised_jit(X, prototypes, proto_labels, distance_fn, n_classes):
    """JIT-compiled supervised probability prediction."""
    distances = distance_fn(X, prototypes)
    class_dists = stratified_min_pooling(distances, proto_labels, n_classes)
    return jax.nn.softmax(-class_dists, axis=1)


@partial(jit, static_argnums=(2,))
def _predict_unsupervised_jit(X, prototypes, distance_fn):
    """JIT-compiled unsupervised prediction (BMU assignment)."""
    distances = distance_fn(X, prototypes)
    return jnp.argmin(distances, axis=1)


@partial(jit, static_argnums=(2,))
def _transform_unsupervised_jit(X, prototypes, distance_fn):
    """JIT-compiled unsupervised distance transform."""
    return distance_fn(X, prototypes)


# --- Supervised Base ---

class SupervisedState(NamedTuple):
    """Base state for gradient-based supervised models.

    Subclasses extend this with model-specific fields (e.g., omega, relevances).
    """
    prototypes: jnp.ndarray      # (n_protos, n_features)
    opt_state: object             # optax optimizer state
    loss: jnp.ndarray             # scalar
    iteration: int
    converged: bool


class ScanTrainState(NamedTuple):
    """State for lax.scan-based supervised training loop."""
    params: dict                  # trainable parameters dict
    opt_state: object             # optax optimizer state
    loss: jnp.ndarray             # current loss (scalar)
    prev_loss: jnp.ndarray        # previous iteration loss
    converged: jnp.ndarray        # boolean convergence flag


class SupervisedPrototypeModel(SerializationMixin, MetadataCollectorMixin, QuantizationMixin, ABC):
    """Base class for supervised prototype-based learning models.

    Provides the fit/predict/save/load infrastructure. Subclasses implement:
    - _init_state(X, y, key) -> state NamedTuple
    - _compute_loss(params, X, y, proto_labels) -> scalar
    - _post_update(params) -> params  (optional, for constraints like normalization)

    Parameters
    ----------
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
    patience : int, optional
        Number of consecutive epochs with no improvement before stopping.
        If None (default), stops after a single non-improving step (epsilon check).
        Requires use_scan=False for true early stopping.
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
        - 'sync_period': int (default 6) — sync every k steps
        - 'slow_step_size': float (default 0.5) — interpolation factor
        Default: None (no lookahead).
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
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training. 'float16' or 'bfloat16'.
        Master weights stay in float32; forward/backward pass runs in lower
        precision for ~2x speed and ~half memory on GPU. Float16 uses static
        loss scaling to prevent gradient underflow. Default: None (disabled).
    gradient_checkpointing : bool, optional
        If True, applies ``jax.checkpoint`` (remat) to the loss function.
        Trades compute for memory by re-computing forward activations during
        the backward pass. Beneficial for deep backbone models (Image,
        Siamese). Default: False.
    devices : list of jax.Device or None, optional
        Devices for data-parallel training. Data is sharded across devices
        along the batch dimension; params and optimizer state are replicated.
        When using mini-batch training, ``batch_size`` must be divisible by
        the number of devices. Default: None (single-device).
    """

    def __init__(
        self,
        n_prototypes_per_class: int = 1,
        max_iter: int = 100,
        lr: float = 0.01,
        epsilon: float = 1e-6,
        random_seed: int = 42,
        distance_fn=None,
        optimizer='adam',
        transfer_fn=None,
        margin: float = 0.0,
        callbacks: list = None,
        use_scan: bool = True,
        batch_size: int | None = None,
        lr_scheduler=None,
        lr_scheduler_kwargs: dict | None = None,
        prototypes_initializer=None,
        patience: int | None = None,
        restore_best: bool = False,
        class_weight=None,
        gradient_accumulation_steps: int | None = None,
        ema_decay: float | None = None,
        freeze_params: list | None = None,
        lookahead: dict | None = None,
        mixed_precision: str | None = None,
        gradient_checkpointing: bool = False,
        devices: list | None = None,
    ):
        if isinstance(n_prototypes_per_class, dict):
            for cls, cnt in n_prototypes_per_class.items():
                if cnt < 1:
                    raise ValueError(
                        f"n_prototypes_per_class for class {cls} must be >= 1, got {cnt}"
                    )
        elif isinstance(n_prototypes_per_class, (list, tuple)):
            for i, cnt in enumerate(n_prototypes_per_class):
                if cnt < 1:
                    raise ValueError(
                        f"n_prototypes_per_class for class {i} must be >= 1, got {cnt}"
                    )
        elif n_prototypes_per_class < 1:
            raise ValueError(
                f"n_prototypes_per_class must be >= 1, got {n_prototypes_per_class}"
            )
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")

        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.n_prototypes_per_class = n_prototypes_per_class
        self.max_iter = max_iter
        self.lr = lr
        self.epsilon = epsilon
        self.random_seed = random_seed
        self.margin = margin
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(random_seed)

        # Distance function
        if distance_fn is None:
            distance_fn = squared_euclidean_distance_matrix
        self.distance_fn = distance_fn

        # Transfer function
        if transfer_fn is None:
            from prosemble.core.activations import identity
            transfer_fn = identity
        self.transfer_fn = transfer_fn

        # Optimizer
        self._optimizer_spec = optimizer
        self._optimizer = self._build_optimizer(optimizer, lr)

        # Learning rate scheduler
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}
        if lr_scheduler is not None:
            self._optimizer = self._apply_lr_scheduler(
                self._optimizer, lr_scheduler
            )

        # Lookahead config (applied in Python training loops)
        self.lookahead = lookahead

        # Gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if gradient_accumulation_steps is not None and gradient_accumulation_steps > 1:
            self._optimizer = optax.MultiSteps(
                self._optimizer, every_k_schedule=gradient_accumulation_steps
            )

        # Parameter freezing (applied in fit() when param keys are known)
        self.freeze_params = freeze_params

        # EMA
        self.ema_decay = ema_decay

        # Mixed precision
        if mixed_precision is not None and mixed_precision not in ('float16', 'bfloat16'):
            raise ValueError(
                f"mixed_precision must be 'float16', 'bfloat16', or None, "
                f"got {mixed_precision!r}"
            )
        self.mixed_precision = mixed_precision
        self._mp_dtype = jnp.dtype(mixed_precision) if mixed_precision else None
        self._loss_scale = 2**15 if mixed_precision == 'float16' else 1.0

        # Gradient checkpointing (remat)
        self.gradient_checkpointing = gradient_checkpointing

        # Multi-device data parallelism
        self.devices = devices
        self._mesh = None

        # Default active optimizer (may be wrapped with masking in fit())
        self._active_optimizer = self._optimizer

        # Prototype initializer
        self.prototypes_initializer = prototypes_initializer

        # Early stopping
        self.patience = patience
        self.restore_best = restore_best

        # Class weighting
        self.class_weight = class_weight

        # Callbacks
        self._callbacks = list(callbacks or [])

        # Training mode
        self.use_scan = use_scan

        # Fitted attributes
        self.prototypes_ = None
        self.prototype_labels_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_iter_ = None
        self.loss_ = None
        self.loss_history_ = None
        self.val_loss_history_ = None
        self.best_loss_ = None

    _base_repr_fields = ('n_prototypes_per_class', 'max_iter', 'lr', 'epsilon')

    def __repr__(self) -> str:
        cls = type(self).__name__
        base = ', '.join(f'{k}={getattr(self, k)!r}' for k in self._base_repr_fields)
        extra = ', '.join(f'{k}={getattr(self, k)!r}' for k in self._all_hyperparams)
        parts = f'{base}, {extra}' if extra else base
        return f'{cls}({parts})'

    def _build_optimizer(self, optimizer, lr):
        """Create optax optimizer from spec.

        Supports string names or pre-built optax GradientTransformations.

        String options (26 optimizers):
            adam, adamw, adamax, adamaxw, adan, adabelief, adadelta,
            adafactor, adagrad, amsgrad, fromage, lamb, lars, lion,
            novograd, radam, rmsprop, rprop, sgd, sign_sgd, signum,
            sm3, yogi, noisy_sgd, lbfgs, dpsgd.
        """
        import optax
        if isinstance(optimizer, str):
            optimizers = {
                # Adam family
                'adam': optax.adam,
                'adamw': optax.adamw,
                'adamax': optax.adamax,
                'adamaxw': optax.adamaxw,
                'adan': optax.adan,
                'adabelief': optax.adabelief,
                'amsgrad': optax.amsgrad,
                'radam': optax.radam,
                'lamb': optax.lamb,
                'lion': optax.lion,
                'novograd': optax.novograd,
                # SGD family
                'sgd': optax.sgd,
                'sign_sgd': optax.sign_sgd,
                'signum': optax.signum,
                'noisy_sgd': optax.noisy_sgd,
                'lars': optax.lars,
                # Adaptive
                'rmsprop': optax.rmsprop,
                'adagrad': optax.adagrad,
                'adadelta': lambda lr: optax.adadelta(learning_rate=lr),
                'adafactor': lambda lr: optax.adafactor(learning_rate=lr),
                'sm3': optax.sm3,
                'yogi': optax.yogi,
                'rprop': optax.rprop,
                'fromage': optax.fromage,
                # Special
                'lbfgs': optax.lbfgs,
                'dpsgd': optax.dpsgd,
            }
            if optimizer in optimizers:
                return optimizers[optimizer](lr)
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. "
                f"Options: {', '.join(sorted(optimizers))}."
            )
        # Assume it's already an optax GradientTransformation
        return optimizer

    def _apply_lr_scheduler(self, optimizer, scheduler):
        """Chain a learning rate scheduler with the optimizer.

        Parameters
        ----------
        optimizer : optax.GradientTransformation
            Base optimizer.
        scheduler : optax.Schedule or str
            Learning rate schedule. String options:
            - 'exponential_decay': exponential decay
            - 'cosine_decay': cosine annealing to 0
            - 'warmup_cosine_decay': linear warmup then cosine decay
            - 'warmup_exponential_decay': linear warmup then exponential decay
            - 'warmup_constant': linear warmup then constant
            - 'polynomial': polynomial decay
            - 'linear': linear decay
            - 'piecewise_constant': step-wise constant schedule
            - 'sgdr': cosine annealing with warm restarts
            Or pass a custom optax.Schedule directly.

        Returns
        -------
        optax.GradientTransformation
        """
        import optax
        kw = self.lr_scheduler_kwargs
        if isinstance(scheduler, str):
            schedule_fn = self._build_schedule(scheduler, kw)
        else:
            schedule_fn = scheduler

        return optax.chain(
            optax.scale_by_adam() if self._optimizer_spec == 'adam' else optax.identity(),
            optax.scale_by_schedule(schedule_fn),
            optax.scale(-1.0),
        )

    def _build_schedule(self, name, kw):
        """Build an optax schedule from a string name and kwargs."""
        import optax
        lr = self.lr
        if name == 'exponential_decay':
            return optax.exponential_decay(
                init_value=lr,
                transition_steps=kw.get('transition_steps', 1),
                decay_rate=kw.get('gamma', kw.get('decay_rate', 0.99)),
            )
        elif name == 'cosine_decay':
            return optax.cosine_decay_schedule(
                init_value=lr,
                decay_steps=kw.get('decay_steps', self.max_iter),
            )
        elif name == 'warmup_cosine_decay':
            return optax.warmup_cosine_decay_schedule(
                init_value=kw.get('init_value', 0.0),
                peak_value=kw.get('peak_value', lr),
                warmup_steps=kw.get('warmup_steps', 10),
                decay_steps=kw.get('decay_steps', self.max_iter),
                end_value=kw.get('end_value', 0.0),
            )
        elif name == 'warmup_exponential_decay':
            return optax.warmup_exponential_decay_schedule(
                init_value=kw.get('init_value', 0.0),
                peak_value=kw.get('peak_value', lr),
                warmup_steps=kw.get('warmup_steps', 10),
                transition_steps=kw.get('transition_steps', 1),
                decay_rate=kw.get('gamma', kw.get('decay_rate', 0.99)),
            )
        elif name == 'warmup_constant':
            return optax.warmup_constant_schedule(
                init_value=kw.get('init_value', 0.0),
                peak_value=kw.get('peak_value', lr),
                warmup_steps=kw.get('warmup_steps', 10),
            )
        elif name == 'polynomial':
            return optax.polynomial_schedule(
                init_value=lr,
                end_value=kw.get('end_value', 0.0),
                power=kw.get('power', 1.0),
                transition_steps=kw.get('transition_steps', self.max_iter),
            )
        elif name == 'linear':
            return optax.linear_schedule(
                init_value=lr,
                end_value=kw.get('end_value', 0.0),
                transition_steps=kw.get('transition_steps', self.max_iter),
            )
        elif name == 'piecewise_constant':
            return optax.piecewise_constant_schedule(
                init_value=lr,
                boundaries_and_scales=kw.get('boundaries_and_scales', {}),
            )
        elif name == 'sgdr':
            # Cosine annealing with warm restarts
            return optax.sgdr_schedule(
                cosine_kwargs=kw.get('cosine_kwargs', [
                    {'init_value': 0.0, 'peak_value': lr, 'warmup_steps': 0,
                     'decay_steps': self.max_iter, 'end_value': 0.0}
                ]),
            )
        else:
            valid = [
                'exponential_decay', 'cosine_decay', 'warmup_cosine_decay',
                'warmup_exponential_decay', 'warmup_constant', 'polynomial',
                'linear', 'piecewise_constant', 'sgdr',
            ]
            raise ValueError(
                f"Unknown scheduler '{name}'. Options: {', '.join(valid)}, "
                "or pass a custom optax.Schedule."
            )

    def _init_prototypes(self, X, y, n_per_class, key):
        """Initialize prototypes using the configured initializer.

        Dispatches to the user-selected initializer (string name or callable).
        If no initializer was set, defaults to 'stratified_random'.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data.
        y : array of shape (n_samples,)
            Training labels.
        n_per_class : int, list, or dict
            Number of prototypes per class.
        key : jax.random.PRNGKey
            Random key.

        Returns
        -------
        prototypes : array of shape (n_prototypes, n_features)
        prototype_labels : array of shape (n_prototypes,)
        """
        init = self.prototypes_initializer
        if init is None:
            return stratified_selection_init(X, y, n_per_class, key)
        if isinstance(init, str):
            if init not in _PROTOTYPE_INITIALIZER_REGISTRY:
                valid = ', '.join(sorted(_PROTOTYPE_INITIALIZER_REGISTRY))
                raise ValueError(
                    f"Unknown prototypes_initializer '{init}'. "
                    f"Options: {valid}, or pass a callable."
                )
            return _PROTOTYPE_INITIALIZER_REGISTRY[init](X, y, n_per_class, key)
        # Callable initializer — call with standard signature
        return init(X, y, n_per_class, key)

    def _init_state(self, X, y, key):
        """Initialize model state. Subclasses override this.

        Parameters
        ----------
        X : array of shape (n, d)
        y : array of shape (n,)
        key : JAX PRNGKey

        Returns
        -------
        state : NamedTuple
            Initial model state.
        params : dict
            Dictionary of trainable parameters.
        proto_labels : array
            Labels for each prototype.
        """
        n_classes = int(jnp.max(y)) + 1
        key1, key2 = jax.random.split(key)
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        params = {'prototypes': prototypes}
        opt_state = self._optimizer.init(params)
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        """Compute differentiable loss. Subclasses override this.

        Parameters
        ----------
        params : dict
            Trainable parameters (must include 'prototypes').
        X : array of shape (n, d)
        y : array of shape (n,)
        proto_labels : array of shape (n_protos,)

        Returns
        -------
        scalar loss
        """
        raise NotImplementedError("Subclasses must implement _compute_loss")

    def _compute_loss_mp(self, params, X, y, proto_labels):
        """Mixed-precision wrapper: cast params down, scale loss."""
        params_mp = jax.tree.map(lambda p: p.astype(self._mp_dtype), params)
        return self._compute_loss(params_mp, X, y, proto_labels) * self._loss_scale

    def _value_and_grad_fn(self, params, X, y, proto_labels):
        """Compute loss and gradients.

        Supports optional mixed precision, gradient checkpointing (remat),
        and custom VJP rules.
        """
        # Select loss function
        if getattr(self, '_use_custom_vjp', False):
            loss_fn = self._custom_vjp_loss
        elif self._mp_dtype is not None:
            loss_fn = self._compute_loss_mp
        else:
            loss_fn = self._compute_loss

        # Apply gradient checkpointing (recompute forward during backward)
        if self.gradient_checkpointing and not getattr(self, '_use_custom_vjp', False):
            loss_fn = jax.checkpoint(loss_fn)

        loss, grads = jax.value_and_grad(loss_fn)(params, X, y, proto_labels)

        # Unscale for mixed precision
        if self._mp_dtype is not None and not getattr(self, '_use_custom_vjp', False):
            loss = loss / self._loss_scale
            grads = jax.tree.map(lambda g: g / self._loss_scale, grads)

        return loss, grads

    def _post_update(self, params):
        """Apply constraints after gradient update (e.g., normalize relevances).

        Default: no-op. Subclasses override as needed.
        """
        return params

    def _custom_vjp_loss(self, params, X, y, proto_labels):
        """Compute loss with custom VJP rule.

        Override this method and set ``self._use_custom_vjp = True`` in
        ``__init__`` to use a custom backward pass via ``@jax.custom_vjp``.

        Default: None (uses standard autodiff on ``_compute_loss``).
        """
        return None

    def _get_weighted_data(self, X, y, sample_weight, key):
        """Apply sample weighting via weighted resampling.

        Resamples the training data with replacement, where each sample's
        probability is proportional to its weight. This is exact for
        stochastic optimization and works with any loss function.
        """
        if sample_weight is None:
            return X, y
        n = X.shape[0]
        probs = sample_weight / jnp.sum(sample_weight)
        indices = jax.random.choice(key, n, shape=(n,), replace=True, p=probs)
        return X[indices], y[indices]

    def _compute_val_loss(self, params, proto_labels):
        """Compute validation loss (no gradients)."""
        if self._val_data is None:
            return None
        X_val, y_val = self._val_data
        return float(self._compute_loss(params, X_val, y_val, proto_labels))

    def _init_lookahead(self, params):
        """Initialize slow params for lookahead."""
        if self.lookahead is None:
            return None
        return jax.tree.map(lambda x: x.copy(), params)

    def _update_lookahead(self, slow_params, fast_params, step):
        """Sync slow params with fast params every sync_period steps."""
        if slow_params is None:
            return None, fast_params
        sync_period = self.lookahead.get('sync_period', 6)
        alpha = self.lookahead.get('slow_step_size', 0.5)
        if (step + 1) % sync_period == 0:
            # Interpolate: slow = slow + alpha * (fast - slow)
            new_slow = jax.tree.map(
                lambda s, f: s + alpha * (f - s), slow_params, fast_params
            )
            return new_slow, new_slow  # fast resets to slow
        return slow_params, fast_params

    def _init_ema(self, params):
        """Initialize EMA shadow parameters."""
        if self.ema_decay is None:
            return None
        return jax.tree.map(lambda x: x.copy(), params)

    def _update_ema(self, ema_params, params):
        """Update EMA: shadow = decay * shadow + (1 - decay) * params."""
        if ema_params is None:
            return None
        decay = self.ema_decay
        return jax.tree.map(
            lambda ema, p: decay * ema + (1.0 - decay) * p,
            ema_params, params,
        )

    def _check_patience(self, loss_history, patience):
        """Check if training should stop based on patience.

        Returns True if the last `patience` epochs showed no improvement.
        """
        if len(loss_history) < patience + 1:
            return False
        best_recent = min(loss_history[-(patience + 1):-1])
        # If current loss hasn't improved over the best in the patience window
        return loss_history[-1] >= best_recent - self.epsilon

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         opt_state=None, **kwargs):
        """Store final results as fitted attributes."""
        best_params = kwargs.get('best_params')
        best_loss = kwargs.get('best_loss')
        val_loss_history = kwargs.get('val_loss_history')

        if self.restore_best and best_params is not None:
            params = best_params

        # Unshard params back to single device after multi-device training
        if self._mesh is not None:
            from prosemble.core.distributed import unshard_params
            params = unshard_params(params)
            proto_labels = jax.device_put(proto_labels, jax.devices()[0])

        self.prototypes_ = params['prototypes']
        self.prototype_labels_ = proto_labels
        self.loss_ = float(loss_history[-1]) if len(loss_history) > 0 else None
        self.loss_history_ = jnp.array(loss_history)
        self.n_iter_ = n_iter
        if opt_state is not None:
            self._opt_state = opt_state
        if val_loss_history is not None:
            self.val_loss_history_ = jnp.array(val_loss_history)
        if best_loss is not None:
            self.best_loss_ = best_loss

    def fit(self, X, y, initial_prototypes=None, initial_labels=None,
            validation_data=None, sample_weight=None, resume=False) -> Self:
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        initial_prototypes : array-like, optional
            Initial prototype positions. For warm-starting from another model.
        initial_labels : array-like, optional
            Labels for initial_prototypes. Required when initial_prototypes
            have a different number than what n_prototypes_per_class produces.
        validation_data : tuple of (X_val, y_val), optional
            Validation data for monitoring. When provided with restore_best=True,
            the model restores params with the lowest validation loss.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights for the loss function.
        resume : bool, default=False
            If True, resume training from the model's current fitted state.
            Uses current prototypes (and other fitted params) as starting point
            with a fresh optimizer. Cannot be combined with initial_prototypes.

        Returns
        -------
        self
        """
        if resume and initial_prototypes is not None:
            raise ValueError("Cannot use both resume=True and initial_prototypes")

        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.int32)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        # Multi-device sharding setup
        if self.devices is not None:
            from prosemble.core.distributed import create_mesh, shard_data
            self._mesh = create_mesh(self.devices)
            n_devices = len(self.devices)
            if self.batch_size is not None and self.batch_size % n_devices != 0:
                raise ValueError(
                    f"batch_size ({self.batch_size}) must be divisible by "
                    f"number of devices ({n_devices}) for data parallelism."
                )
            X, y = shard_data(X, y, self._mesh)

        self.classes_ = jnp.unique(y)
        self.n_classes_ = int(len(self.classes_))

        # Compute sample weights from class_weight if provided
        if sample_weight is not None:
            self._sample_weight = jnp.asarray(sample_weight, dtype=jnp.float32)
        elif self.class_weight is not None:
            self._sample_weight = self._compute_sample_weights(y)
        else:
            self._sample_weight = None

        # Validation data
        if validation_data is not None:
            X_val, y_val = validation_data
            self._val_data = (
                jnp.asarray(X_val, dtype=jnp.float32),
                jnp.asarray(y_val, dtype=jnp.int32),
            )
        else:
            self._val_data = None

        # Initialize
        if resume:
            self._check_fitted()
            params = {'prototypes': self.prototypes_}
            proto_labels = self.prototype_labels_
            params = self._get_resume_params(params)
        else:
            state, params, proto_labels = self._init_state(X, y, self.key)
            if initial_prototypes is not None:
                params['prototypes'] = jnp.asarray(initial_prototypes, dtype=jnp.float32)
                if initial_labels is not None:
                    proto_labels = jnp.asarray(initial_labels, dtype=jnp.int32)

        # Apply parameter freezing if requested
        optimizer = self._optimizer
        if self.freeze_params is not None:
            freeze_set = set(self.freeze_params)
            mask = {k: (k not in freeze_set) for k in params}
            optimizer = optax.masked(optimizer, mask)

        # Store active optimizer (possibly with masking)
        self._active_optimizer = optimizer

        # Reuse optimizer state on resume, otherwise initialize fresh
        if resume and hasattr(self, '_opt_state') and self._opt_state is not None:
            fresh = self._active_optimizer.init(params)
            saved_n = len(jax.tree.leaves(self._opt_state))
            fresh_n = len(jax.tree.leaves(fresh))
            if saved_n == fresh_n:
                opt_state = self._opt_state
            else:
                opt_state = fresh
        else:
            opt_state = self._active_optimizer.init(params)

        # Replicate params and optimizer state across devices
        if self._mesh is not None:
            from prosemble.core.distributed import replicate_params, replicate_opt_state
            params = replicate_params(params, self._mesh)
            opt_state = replicate_opt_state(opt_state, self._mesh)

        if self._callbacks:
            return self._fit_with_callbacks(X, y, params, opt_state, proto_labels)
        else:
            return self._fit_loop(X, y, params, opt_state, proto_labels)

    def partial_fit(self, X, y) -> Self:
        """Perform a single gradient update on the given data.

        For incremental/online learning. The model must already be fitted
        via ``fit()``. Optimizer state is preserved across calls.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.int32)

        # Shard data if multi-device
        if self._mesh is not None:
            from prosemble.core.distributed import shard_data, replicate_params, replicate_opt_state
            X, y = shard_data(X, y, self._mesh)

        # Build params from fitted state
        params = self._get_resume_params({'prototypes': self.prototypes_})
        proto_labels = self.prototype_labels_

        # Initialize optimizer if not yet available
        if not hasattr(self, '_active_optimizer') or self._active_optimizer is None:
            self._active_optimizer = self._optimizer
        if not hasattr(self, '_opt_state') or self._opt_state is None:
            self._opt_state = self._active_optimizer.init(params)

        # Replicate params, optimizer state, and proto_labels for multi-device
        if self._mesh is not None:
            params = replicate_params(params, self._mesh)
            self._opt_state = replicate_opt_state(self._opt_state, self._mesh)
            proto_labels = replicate_params({'_': proto_labels}, self._mesh)['_']

        # Single gradient step
        loss, grads = self._value_and_grad_fn(params, X, y, proto_labels)
        updates, self._opt_state = self._active_optimizer.update(
            grads, self._opt_state, params
        )
        params = optax.apply_updates(params, updates)
        params = self._post_update(params)

        # Store updated params
        self._extract_results(
            params, proto_labels, [float(loss)], self.n_iter_ + 1,
            opt_state=self._opt_state,
        )
        return self

    def _get_resume_params(self, params):
        """Build params dict from fitted state for resume.

        Default returns just prototypes. Subclasses with additional learnable
        parameters (omega, relevances, backbone, etc.) should override to add
        their parameters to the dict.

        Parameters
        ----------
        params : dict
            Dict with 'prototypes' already set from self.prototypes_.

        Returns
        -------
        dict
            Updated params dict.
        """
        return params

    def _compute_sample_weights(self, y):
        """Compute per-sample weights from class_weight spec."""
        n_samples = len(y)
        classes = jnp.unique(y)
        n_classes = len(classes)

        if self.class_weight == 'balanced':
            # Inversely proportional to class frequency
            weight_map = {}
            for c in classes:
                c_int = int(c)
                n_c = int(jnp.sum(y == c))
                weight_map[c_int] = n_samples / (n_classes * n_c)
        elif isinstance(self.class_weight, dict):
            weight_map = self.class_weight
        else:
            raise ValueError(
                f"class_weight must be 'balanced' or dict, got {type(self.class_weight)}"
            )

        weights = jnp.ones(n_samples, dtype=jnp.float32)
        for c, w in weight_map.items():
            weights = jnp.where(y == c, w, weights)
        return weights

    @partial(jit, static_argnums=(0,))
    def _training_step_scan(self, state, X, y, proto_labels):
        """Single JIT-compiled training step for lax.scan."""
        params, opt_state, loss, prev_loss, converged = state

        # Compute loss and gradients
        loss_val, grads = self._value_and_grad_fn(params, X, y, proto_labels)

        # Apply optimizer update
        updates, new_opt_state = self._active_optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Apply constraints
        new_params = self._post_update(new_params)

        # Check convergence
        has_converged = converged | (jnp.abs(loss_val - prev_loss) < self.epsilon)

        # Freeze params when converged (keep old values)
        frozen_params = jax.tree.map(
            lambda new, old: jnp.where(converged, old, new),
            new_params, params
        )
        frozen_opt_state = jax.tree.map(
            lambda new, old: jnp.where(converged, old, new),
            new_opt_state, opt_state
        )
        frozen_loss = jnp.where(converged, loss, loss_val)

        new_state = ScanTrainState(
            params=frozen_params,
            opt_state=frozen_opt_state,
            loss=frozen_loss,
            prev_loss=loss_val,
            converged=has_converged,
        )
        return new_state, frozen_loss

    @partial(jit, static_argnums=(0,))
    def _fit_scan(self, X, y, params, opt_state, proto_labels):
        """Scan-based training loop (JIT-compiled, no callbacks)."""
        initial_state = ScanTrainState(
            params=params,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            prev_loss=jnp.array(float('inf')),
            converged=jnp.array(False),
        )

        def scan_fn(state, _):
            return self._training_step_scan(state, X, y, proto_labels)

        final_state, loss_history = lax.scan(
            scan_fn, initial_state, None, length=self.max_iter
        )
        return final_state, loss_history

    def _fit_loop(self, X, y, params, opt_state, proto_labels):
        """Training loop without callbacks. Routes based on use_scan and batch_size."""
        if self.batch_size is not None:
            if self.use_scan:
                return self._fit_minibatch_scan(X, y, params, opt_state, proto_labels)
            else:
                return self._fit_minibatch_python(X, y, params, opt_state, proto_labels)
        if self.use_scan:
            return self._fit_loop_scan(X, y, params, opt_state, proto_labels)
        else:
            return self._fit_loop_python(X, y, params, opt_state, proto_labels)

    def _fit_loop_scan(self, X, y, params, opt_state, proto_labels):
        """lax.scan training: JIT-compiled, runs all max_iter iterations."""
        final_state, loss_history = self._fit_scan(
            X, y, params, opt_state, proto_labels
        )

        # Find actual convergence iteration
        converged_mask = jnp.abs(jnp.diff(loss_history)) < self.epsilon
        first_converged = jnp.argmax(converged_mask)
        has_any_converged = jnp.any(converged_mask)
        n_iter = jnp.where(has_any_converged, first_converged + 2, self.max_iter)

        self._extract_results(
            final_state.params, proto_labels,
            loss_history.tolist(), int(n_iter),
            opt_state=final_state.opt_state,
        )
        return self

    def _fit_loop_python(self, X, y, params, opt_state, proto_labels):
        """Python for-loop training: true early stopping, no wasted compute."""
        loss_history = []
        val_loss_history = []
        best_params = None
        best_loss = float('inf')
        patience = self.patience
        key = self.key
        ema_params = self._init_ema(params)
        slow_params = self._init_lookahead(params)

        for i in range(self.max_iter):
            # Apply sample weighting via resampling
            if self._sample_weight is not None:
                key, subkey = jax.random.split(key)
                X_step, y_step = self._get_weighted_data(
                    X, y, self._sample_weight, subkey
                )
            else:
                X_step, y_step = X, y

            loss_val, grads = self._value_and_grad_fn(
                params, X_step, y_step, proto_labels
            )
            loss_history.append(float(loss_val))

            updates, opt_state = self._active_optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._post_update(params)
            slow_params, params = self._update_lookahead(slow_params, params, i)
            ema_params = self._update_ema(ema_params, params)

            # Validation loss
            val_loss = self._compute_val_loss(params, proto_labels)
            if val_loss is not None:
                val_loss_history.append(val_loss)

            # Track best model (use val loss if available, else train loss)
            monitor_loss = val_loss if val_loss is not None else float(loss_val)
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                if self.restore_best:
                    best_params = jax.tree.map(lambda x: x.copy(), params)

            # Early stopping
            monitor_history = val_loss_history if val_loss_history else loss_history
            if patience is not None:
                if self._check_patience(monitor_history, patience):
                    break
            elif i > 0 and abs(loss_history[-1] - loss_history[-2]) < self.epsilon:
                break

        # Use slow params from lookahead for inference (they generalize better)
        final_params = slow_params if slow_params is not None else params
        if ema_params is not None and not (self.restore_best and best_params is not None):
            final_params = ema_params

        self._extract_results(
            final_params, proto_labels, loss_history, i + 1,
            opt_state=opt_state,
            val_loss_history=val_loss_history or None,
            best_params=best_params,
            best_loss=best_loss,
        )
        return self

    def _fit_minibatch_scan(self, X, y, params, opt_state, proto_labels):
        """Mini-batch training with lax.scan (JIT-compiled).

        Outer scan over epochs, inner scan over mini-batches.
        Per-epoch shuffling uses jax.random.fold_in for deterministic keys.
        """
        n_samples = X.shape[0]
        batch_size = self.batch_size
        n_batches = (n_samples + batch_size - 1) // batch_size
        padded_size = n_batches * batch_size

        # Pad data to be divisible by batch_size
        if padded_size > n_samples:
            pad_n = padded_size - n_samples
            X = jnp.concatenate([X, X[:pad_n]], axis=0)
            y = jnp.concatenate([y, y[:pad_n]], axis=0)

        base_key = self.key

        initial_state = ScanTrainState(
            params=params,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            prev_loss=jnp.array(float('inf')),
            converged=jnp.array(False),
        )

        @partial(jit, static_argnums=())
        def _run_scan(initial_state, X_pad, y_pad):
            def epoch_fn(state, epoch_idx):
                # Per-epoch shuffle
                key = jax.random.fold_in(base_key, epoch_idx)
                perm = jax.random.permutation(key, padded_size)
                X_shuffled = X_pad[perm]
                y_shuffled = y_pad[perm]

                # Reshape into batches
                X_batches = X_shuffled.reshape(n_batches, batch_size, -1)
                y_batches = y_shuffled.reshape(n_batches, batch_size)

                # Inner scan over mini-batches
                def batch_step(carry, batch_data):
                    params_b, opt_state_b = carry
                    X_b, y_b = batch_data
                    loss_val, grads = self._value_and_grad_fn(
                        params_b, X_b, y_b, proto_labels
                    )
                    updates, new_opt = self._active_optimizer.update(
                        grads, opt_state_b, params_b
                    )
                    new_params = optax.apply_updates(params_b, updates)
                    new_params = self._post_update(new_params)
                    return (new_params, new_opt), loss_val

                (new_params, new_opt_state), batch_losses = lax.scan(
                    batch_step,
                    (state.params, state.opt_state),
                    (X_batches, y_batches),
                )
                epoch_loss = jnp.mean(batch_losses)

                # Convergence + freeze
                has_converged = state.converged | (
                    jnp.abs(epoch_loss - state.prev_loss) < self.epsilon
                )
                frozen_params = jax.tree.map(
                    lambda new, old: jnp.where(state.converged, old, new),
                    new_params, state.params,
                )
                frozen_opt = jax.tree.map(
                    lambda new, old: jnp.where(state.converged, old, new),
                    new_opt_state, state.opt_state,
                )
                frozen_loss = jnp.where(state.converged, state.loss, epoch_loss)

                new_state = ScanTrainState(
                    params=frozen_params,
                    opt_state=frozen_opt,
                    loss=frozen_loss,
                    prev_loss=epoch_loss,
                    converged=has_converged,
                )
                return new_state, frozen_loss

            return lax.scan(epoch_fn, initial_state, jnp.arange(self.max_iter))

        final_state, loss_history = _run_scan(initial_state, X, y)

        # Find convergence iteration
        converged_mask = jnp.abs(jnp.diff(loss_history)) < self.epsilon
        first_converged = jnp.argmax(converged_mask)
        has_any = jnp.any(converged_mask)
        n_iter = jnp.where(has_any, first_converged + 2, self.max_iter)

        self._extract_results(
            final_state.params, proto_labels,
            loss_history.tolist(), int(n_iter),
            opt_state=final_state.opt_state,
        )
        return self

    def _fit_minibatch_python(self, X, y, params, opt_state, proto_labels):
        """Mini-batch training with Python for-loop (true early stopping)."""
        n_samples = X.shape[0]
        batch_size = self.batch_size
        key = self.key
        loss_history = []
        val_loss_history = []
        best_params = None
        best_loss = float('inf')
        patience = self.patience
        sw = self._sample_weight
        ema_params = self._init_ema(params)
        slow_params = self._init_lookahead(params)

        for epoch in range(self.max_iter):
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, n_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            sw_shuffled = sw[perm] if sw is not None else None

            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Apply sample weighting via resampling within batch
                if sw_shuffled is not None:
                    key, wkey = jax.random.split(key)
                    batch_w = sw_shuffled[start:end]
                    X_batch, y_batch = self._get_weighted_data(
                        X_batch, y_batch, batch_w, wkey
                    )

                loss_val, grads = self._value_and_grad_fn(
                    params, X_batch, y_batch, proto_labels
                )
                epoch_losses.append(float(loss_val))

                updates, opt_state = self._active_optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                params = self._post_update(params)

            slow_params, params = self._update_lookahead(slow_params, params, epoch)
            ema_params = self._update_ema(ema_params, params)
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            loss_history.append(avg_loss)

            # Validation loss
            val_loss = self._compute_val_loss(params, proto_labels)
            if val_loss is not None:
                val_loss_history.append(val_loss)

            # Track best model
            monitor_loss = val_loss if val_loss is not None else avg_loss
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                if self.restore_best:
                    best_params = jax.tree.map(lambda x: x.copy(), params)

            # Early stopping
            monitor_history = val_loss_history if val_loss_history else loss_history
            if patience is not None:
                if self._check_patience(monitor_history, patience):
                    break
            elif epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < self.epsilon:
                break

        final_params = slow_params if slow_params is not None else params
        if ema_params is not None and not (self.restore_best and best_params is not None):
            final_params = ema_params

        self._extract_results(
            final_params, proto_labels, loss_history, epoch + 1,
            opt_state=opt_state,
            val_loss_history=val_loss_history or None,
            best_params=best_params,
            best_loss=best_loss,
        )
        return self

    def _fit_with_callbacks(self, X, y, params, opt_state, proto_labels):
        """Training loop with callback support."""
        self._notify_fit_start(X)
        loss_history = []
        val_loss_history = []
        best_params = None
        best_loss = float('inf')
        patience = self.patience
        key = self.key
        ema_params = self._init_ema(params)
        slow_params = self._init_lookahead(params)

        for i in range(self.max_iter):
            # Apply sample weighting
            if self._sample_weight is not None:
                key, subkey = jax.random.split(key)
                X_step, y_step = self._get_weighted_data(
                    X, y, self._sample_weight, subkey
                )
            else:
                X_step, y_step = X, y

            loss_val, grads = self._value_and_grad_fn(
                params, X_step, y_step, proto_labels
            )
            loss_history.append(float(loss_val))

            updates, opt_state = self._active_optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._post_update(params)
            slow_params, params = self._update_lookahead(slow_params, params, i)
            ema_params = self._update_ema(ema_params, params)

            # Validation loss
            val_loss = self._compute_val_loss(params, proto_labels)
            if val_loss is not None:
                val_loss_history.append(val_loss)

            # Track best model
            monitor_loss = val_loss if val_loss is not None else float(loss_val)
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                if self.restore_best:
                    best_params = jax.tree.map(lambda x: x.copy(), params)

            # Notify callbacks
            info = self._build_info(params, proto_labels, i, loss_val)
            if val_loss is not None:
                info['val_loss'] = val_loss
            self._notify_iteration(info)

            # Early stopping
            monitor_history = val_loss_history if val_loss_history else loss_history
            if patience is not None:
                if self._check_patience(monitor_history, patience):
                    break
            elif i > 0 and abs(loss_history[-1] - loss_history[-2]) < self.epsilon:
                break

        final_params = slow_params if slow_params is not None else params
        if ema_params is not None and not (self.restore_best and best_params is not None):
            final_params = ema_params

        self._extract_results(
            final_params, proto_labels, loss_history, i + 1,
            opt_state=opt_state,
            val_loss_history=val_loss_history or None,
            best_params=best_params,
            best_loss=best_loss,
        )
        info = self._build_info(final_params, proto_labels, i, loss_history[-1])
        self._notify_fit_end(info)
        return self

    def _build_info(self, params, proto_labels, iteration, loss_val):
        """Build info dict for callbacks."""
        return {
            'prototypes': params['prototypes'],
            'prototype_labels': proto_labels,
            'iteration': iteration,
            'loss': float(loss_val),
            'max_iter': self.max_iter,
        }

    def _get_quantizable_attrs(self) -> list[str]:
        """Return list of attr names for quantizable parameters.

        Subclasses override to add model-specific parameters (omega, relevances, etc.).
        """
        attrs = []
        if self.prototypes_ is not None:
            attrs.append('prototypes_')
        return attrs

    def predict(self, X):
        """Predict class labels via Winner-Takes-All Competition.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_supervised_jit(
            X, self.prototypes_, self.prototype_labels_, self.distance_fn
        )

    def predict_proba(self, X):
        """Predict class probabilities via softmin of distances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_proba_supervised_jit(
            X, self.prototypes_, self.prototype_labels_,
            self.distance_fn, self.n_classes_
        )

    def prototype_win_ratios(self, X, y):
        """Compute how often each prototype wins on correctly classified samples.

        For each sample, find the closest prototype. A prototype "wins" when it
        is the nearest prototype and its label matches the sample label. The win
        ratio is the fraction of total samples each prototype wins on.

        Useful for identifying "dead" prototypes that never win, which may
        be candidates for removal or reinitialization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        win_ratios : array of shape (n_prototypes,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.int32)
        distances = self.distance_fn(X, self.prototypes_)
        nearest = jnp.argmin(distances, axis=1)
        nearest_labels = self.prototype_labels_[nearest]
        correct = (nearest_labels == y)
        n_protos = self.prototypes_.shape[0]
        wins = jnp.zeros(n_protos)
        for j in range(n_protos):
            wins = wins.at[j].set(jnp.sum((nearest == j) & correct))
        return wins / len(X)

    def export_predict(self, batch_size=1):
        """Export JIT-compiled predict function for deployment.

        Returns a jax.export.Exported object that can be serialized
        with `exported.serialize()` and later deserialized with
        `jax.export.deserialize()` for inference without the model.

        Parameters
        ----------
        batch_size : int
            Fixed input batch size for the exported function.

        Returns
        -------
        exported : jax.export.Exported
            Portable compiled prediction function.

        Examples
        --------
        >>> model.fit(X_train, y_train)
        >>> exported = model.export_predict(batch_size=32)
        >>> blob = exported.serialize()  # bytes
        >>> # Later, without the model:
        >>> loaded = jax.export.deserialize(blob)
        >>> preds = loaded.call(X_batch)
        """
        from jax import export as jax_export

        self._check_fitted()
        n_features = self.prototypes_.shape[1]

        prototypes = self.prototypes_
        proto_labels = self.prototype_labels_
        distance_fn = self.distance_fn

        @jit
        def predict_fn(X):
            distances = distance_fn(X, prototypes)
            return wtac(distances, proto_labels)

        input_spec = jax.ShapeDtypeStruct(
            (batch_size, n_features), jnp.float32
        )
        return jax_export.export(predict_fn)(input_spec)

    def export_onnx(self, batch_size=1, opset_version=17, path=None):
        """Export predict function to ONNX format.

        Parameters
        ----------
        batch_size : int
            Fixed input batch size. Use -1 for dynamic batch.
        opset_version : int
            ONNX opset version. Default: 17.
        path : str, optional
            If provided, save ONNX model to this file path.

        Returns
        -------
        onnx.ModelProto
            The exported ONNX model.

        Raises
        ------
        NotImplementedError
            If the model's distance function is not supported.
        """
        from prosemble.core.onnx_export import export_onnx
        return export_onnx(self, batch_size, opset_version, path)

    def _check_fitted(self):
        if self.prototypes_ is None:
            raise NotFittedError("Model not fitted yet. Call fit() first.")

    # --- Serialization ---

    def _get_hyperparams(self):
        params = {
            'n_prototypes_per_class': self.n_prototypes_per_class,
            'max_iter': self.max_iter,
            'lr': self.lr,
            'epsilon': self.epsilon,
            'random_seed': self.random_seed,
            'margin': self.margin,
            'mixed_precision': self.mixed_precision,
            'prototypes_initializer': self.prototypes_initializer if isinstance(
                self.prototypes_initializer, str
            ) else None,
        }
        for name in self._all_hyperparams:
            val = getattr(self, name)
            params[name] = float(val) if hasattr(val, 'item') else val
        return params

    def _get_fitted_arrays(self):
        arrays = {}
        if self.prototypes_ is not None:
            arrays['prototypes_'] = np.asarray(self.prototypes_)
        if self.prototype_labels_ is not None:
            arrays['prototype_labels_'] = np.asarray(self.prototype_labels_)
        if self.loss_history_ is not None:
            arrays['loss_history_'] = np.asarray(self.loss_history_)
        for name in self._all_fitted_array_names:
            val = getattr(self, name, None)
            if val is not None:
                arrays[name] = np.asarray(val)
        return arrays

    def _set_fitted_arrays(self, arrays):
        if 'prototypes_' in arrays:
            self.prototypes_ = jnp.asarray(arrays['prototypes_'])
        if 'prototype_labels_' in arrays:
            self.prototype_labels_ = jnp.asarray(arrays['prototype_labels_'])
        if 'loss_history_' in arrays:
            self.loss_history_ = jnp.asarray(arrays['loss_history_'])
        for name in self._all_fitted_array_names:
            if name in arrays:
                setattr(self, name, jnp.asarray(arrays[name]))

    # --- SerializationMixin hooks ---

    def _get_save_metadata(self):
        return {
            'n_iter_': self.n_iter_,
            'loss_': self.loss_,
            'n_classes_': self.n_classes_,
        }

    def _restore_metadata(self, metadata):
        self.n_iter_ = metadata.get('n_iter_')
        self.loss_ = metadata.get('loss_')
        self.n_classes_ = metadata.get('n_classes_')
        if self.prototype_labels_ is not None:
            self.classes_ = jnp.unique(self.prototype_labels_)

    def _save_optimizer_state(self, arrays, metadata):
        if hasattr(self, '_opt_state') and self._opt_state is not None:
            opt_leaves = jax.tree.leaves(self._opt_state)
            for i, leaf in enumerate(opt_leaves):
                arrays[f'__opt_state_{i}__'] = np.asarray(leaf)
            metadata['opt_state_n_leaves'] = len(opt_leaves)

    def _load_optimizer_state(self, data, metadata):
        n_leaves = metadata.get('opt_state_n_leaves')
        if n_leaves is not None and n_leaves > 0:
            saved_leaves = [
                jnp.asarray(data[f'__opt_state_{i}__'])
                for i in range(n_leaves)
            ]
            params = self._get_resume_params({'prototypes': self.prototypes_})
            optimizer = self._optimizer
            if self.freeze_params is not None:
                freeze_set = set(self.freeze_params)
                mask = {k: (k not in freeze_set) for k in params}
                optimizer = optax.masked(optimizer, mask)
            skeleton = optimizer.init(params)
            treedef = jax.tree.structure(skeleton)
            if len(saved_leaves) == len(jax.tree.leaves(skeleton)):
                self._opt_state = jax.tree.unflatten(treedef, saved_leaves)

    # --- Callback notifications ---

    def _notify_fit_start(self, X):
        for cb in self._callbacks:
            cb.on_fit_start(self, X)

    def _notify_iteration(self, info):
        for cb in self._callbacks:
            cb.on_iteration_end(self, info)

    def _notify_fit_end(self, info):
        for cb in self._callbacks:
            cb.on_fit_end(self, info)


# --- Unsupervised Base ---

class UnsupervisedPrototypeModel(SerializationMixin, MetadataCollectorMixin, QuantizationMixin, ABC):
    """Base class for unsupervised prototype-based topology models.

    For NeuralGas, GrowingNeuralGas, KohonenSOM.

    Parameters
    ----------
    n_prototypes : int
        Number of prototypes/nodes.
    max_iter : int
        Maximum training iterations.
    lr : float
        Initial learning rate.
    epsilon : float
        Convergence threshold.
    random_seed : int
        Random seed.
    distance_fn : callable, optional
        Distance function.
    callbacks : list, optional
        Callback objects.
    use_scan : bool
        If True (default), use jax.lax.scan for training (faster, JIT-compiled,
        but runs all max_iter iterations even after convergence).
        If False, use a Python for-loop with true early stopping.
    patience : int, optional
        Epochs with no improvement before early stopping. Default: None.
    restore_best : bool
        If True, restore parameters from the lowest-loss epoch. Default: False.
    """

    def __init__(
        self,
        n_prototypes: int,
        max_iter: int = 100,
        lr: float = 0.01,
        epsilon: float = 1e-6,
        random_seed: int = 42,
        distance_fn=None,
        callbacks: list = None,
        use_scan: bool = True,
        patience: int = None,
        restore_best: bool = False,
    ):
        if n_prototypes < 1:
            raise ValueError(f"n_prototypes must be >= 1, got {n_prototypes}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if patience is not None and patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.n_prototypes = n_prototypes
        self.max_iter = max_iter
        self.lr = lr
        self.epsilon = epsilon
        self.random_seed = random_seed
        self.key = jax.random.PRNGKey(random_seed)

        if distance_fn is None:
            distance_fn = squared_euclidean_distance_matrix
        self.distance_fn = distance_fn

        self._callbacks = list(callbacks or [])
        self.use_scan = use_scan
        self.patience = patience
        self.restore_best = restore_best

        # Fitted attributes
        self.prototypes_ = None
        self.n_iter_ = None
        self.loss_ = None
        self.loss_history_ = None
        self.best_loss_ = None

    _base_repr_fields = ('n_prototypes', 'max_iter', 'lr', 'epsilon')

    def __repr__(self) -> str:
        cls = type(self).__name__
        base = ', '.join(f'{k}={getattr(self, k)!r}' for k in self._base_repr_fields)
        extra = ', '.join(f'{k}={getattr(self, k)!r}' for k in self._all_hyperparams)
        parts = f'{base}, {extra}' if extra else base
        return f'{cls}({parts})'

    def _check_fitted(self):
        if self.prototypes_ is None:
            raise NotFittedError("Model not fitted yet. Call fit() first.")

    @abstractmethod
    def fit(self, X):
        """Fit the model to data X."""
        ...

    def _check_patience(self, objective_history, patience):
        """Check if training should stop due to lack of improvement."""
        if len(objective_history) <= patience:
            return False
        best = min(objective_history[:-patience])
        recent = objective_history[-patience:]
        return all(r >= best for r in recent)

    def _get_quantizable_attrs(self) -> list[str]:
        """Return list of attr names for quantizable parameters."""
        attrs = []
        if self.prototypes_ is not None:
            attrs.append('prototypes_')
        return attrs

    def predict(self, X):
        """Assign each sample to closest prototype (BMU)."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_unsupervised_jit(X, self.prototypes_, self.distance_fn)

    def transform(self, X):
        """Return distance matrix to all prototypes."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _transform_unsupervised_jit(X, self.prototypes_, self.distance_fn)

    def export_predict(self, batch_size=1):
        """Export JIT-compiled predict function for deployment.

        Parameters
        ----------
        batch_size : int
            Fixed input batch size.

        Returns
        -------
        exported : jax.export.Exported
        """
        from jax import export as jax_export

        self._check_fitted()
        n_features = self.prototypes_.shape[1]
        prototypes = self.prototypes_
        distance_fn = self.distance_fn

        @jit
        def predict_fn(X):
            distances = distance_fn(X, prototypes)
            return jnp.argmin(distances, axis=1)

        input_spec = jax.ShapeDtypeStruct(
            (batch_size, n_features), jnp.float32
        )
        return jax_export.export(predict_fn)(input_spec)

    def export_onnx(self, batch_size=1, opset_version=17, path=None):
        """Export predict function to ONNX format.

        Parameters
        ----------
        batch_size : int
            Fixed input batch size. Use -1 for dynamic batch.
        opset_version : int
            ONNX opset version. Default: 17.
        path : str, optional
            If provided, save ONNX model to this file path.

        Returns
        -------
        onnx.ModelProto
        """
        from prosemble.core.onnx_export import export_onnx
        return export_onnx(self, batch_size, opset_version, path)

    # --- Serialization ---

    def _get_hyperparams(self):
        params = {
            'n_prototypes': self.n_prototypes,
            'max_iter': self.max_iter,
            'lr': self.lr,
            'epsilon': self.epsilon,
            'random_seed': self.random_seed,
        }
        for name in self._all_hyperparams:
            val = getattr(self, name)
            params[name] = float(val) if hasattr(val, 'item') else val
        return params

    def _get_fitted_arrays(self):
        arrays = {}
        if self.prototypes_ is not None:
            arrays['prototypes_'] = np.asarray(self.prototypes_)
        if self.loss_history_ is not None:
            arrays['loss_history_'] = np.asarray(self.loss_history_)
        for name in self._all_fitted_array_names:
            val = getattr(self, name, None)
            if val is not None:
                arrays[name] = np.asarray(val)
        return arrays

    def _set_fitted_arrays(self, arrays):
        if 'prototypes_' in arrays:
            self.prototypes_ = jnp.asarray(arrays['prototypes_'])
        if 'loss_history_' in arrays:
            self.loss_history_ = jnp.asarray(arrays['loss_history_'])
        for name in self._all_fitted_array_names:
            if name in arrays:
                setattr(self, name, jnp.asarray(arrays[name]))

    # --- SerializationMixin hooks ---

    def _get_save_metadata(self):
        return {
            'n_iter_': self.n_iter_,
            'loss_': self.loss_,
        }

    def _restore_metadata(self, metadata):
        self.n_iter_ = metadata.get('n_iter_')
        self.loss_ = metadata.get('loss_')

    # --- Callback notifications ---

    def _notify_fit_start(self, X):
        for cb in self._callbacks:
            cb.on_fit_start(self, X)

    def _notify_iteration(self, info):
        for cb in self._callbacks:
            cb.on_iteration_end(self, info)

    def _notify_fit_end(self, info):
        for cb in self._callbacks:
            cb.on_fit_end(self, info)
