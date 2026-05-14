"""
Learning Vector Quantization 2.1 (LVQ2.1).

Non-gradient algorithm that simultaneously updates the closest
same-class and closest different-class prototypes.

References
----------
.. [1] Kohonen, T. (1990). Improved Versions of Learning Vector
       Quantization. IJCNN.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel, NotFittedError
from prosemble.core.distance import squared_euclidean_distance_matrix
from prosemble.core.competitions import wtac
from prosemble.core.losses import _get_dp_dm_with_indices


class LVQ21(SupervisedPrototypeModel):
    """Learning Vector Quantization 2.1.

    For each sample:

    - Find closest same-class prototype :math:`w^+` and closest different-class :math:`w^-`
    - :math:`w^+ \\leftarrow w^+ + \\eta (x - w^+)` (attract :math:`w^+`)
    - :math:`w^- \\leftarrow w^- - \\eta (x - w^-)` (repel :math:`w^-`)

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
        - 'sync_period': int (default 6) — sync every k steps
        - 'slow_step_size': float (default 0.5) — interpolation factor
        Default: None (no lookahead).
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training. 'float16' or 'bfloat16'.
        Master weights stay in float32; forward/backward pass runs in lower
        precision for ~2x speed and ~half memory on GPU. Float16 uses static
        loss scaling to prevent gradient underflow. Default: None (disabled).
    """

    def __init__(self, n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None, mixed_precision=None):
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

    def fit(self, X, y, initial_prototypes=None):
        """Fit LVQ2.1 using competitive learning."""
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.int32)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        self.classes_ = jnp.unique(y)
        self.n_classes_ = int(len(self.classes_))

        key = self.key
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key
        )

        if initial_prototypes is not None:
            prototypes = jnp.asarray(initial_prototypes, dtype=jnp.float32)

        loss_history = []

        for i in range(self.max_iter):
            distances = squared_euclidean_distance_matrix(X, prototypes)

            # Get d+, d-, and their indices
            dp, dm, wp, wm = _get_dp_dm_with_indices(distances, y, proto_labels)

            # Update w+: attract toward x
            diff_p = X - prototypes[wp]  # (n, d)
            # Update w-: repel from x
            diff_m = X - prototypes[wm]  # (n, d)

            # Accumulate updates
            n_protos = prototypes.shape[0]
            updates = jnp.zeros_like(prototypes)
            counts = jnp.zeros(n_protos)

            for k in range(n_protos):
                # Attract updates (from w+)
                mask_p = (wp == k)
                attract = jnp.sum(diff_p * mask_p[:, None], axis=0)

                # Repel updates (from w-)
                mask_m = (wm == k)
                repel = jnp.sum(diff_m * mask_m[:, None], axis=0)

                total_update = self.lr * attract - self.lr * repel
                total_count = jnp.sum(mask_p) + jnp.sum(mask_m)
                updates = updates.at[k].set(total_update)
                counts = counts.at[k].set(total_count)

            safe_counts = jnp.maximum(counts, 1.0)
            prototypes = prototypes + updates / safe_counts[:, None]

            # Track loss: mean(d+ - d-)
            loss = float(jnp.mean(dp - dm))
            loss_history.append(loss)

            if i > 0 and abs(loss_history[-1] - loss_history[-2]) < self.epsilon:
                break

        self.prototypes_ = prototypes
        self.prototype_labels_ = proto_labels
        self.loss_ = loss_history[-1]
        self.loss_history_ = jnp.array(loss_history)
        self.n_iter_ = i + 1
        return self
