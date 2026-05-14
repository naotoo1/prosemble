"""
Median LVQ.

A combinatorial optimization approach where prototypes are restricted
to be actual data points. Uses an EM-like alternation between
soft assignment and prototype selection.

References
----------
.. [1] Nebel, D., Hammer, B., & Villmann, T. (2015). Median
       variants of learning vector quantization for learning of
       dissimilarity data.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel, NotFittedError
from prosemble.core.distance import squared_euclidean_distance_matrix
from prosemble.core.competitions import wtac


class MedianLVQ(SupervisedPrototypeModel):
    """Median Learning Vector Quantization.

    Prototypes are always actual data points. The algorithm alternates:

    1. E-step: compute soft assignments (GLVQ-like weights)
    2. M-step: for each prototype, find the data point that minimizes
       the weighted sum of distances

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
        """Fit MedianLVQ."""
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

        for iteration in range(self.max_iter):
            distances = squared_euclidean_distance_matrix(X, prototypes)

            # E-step: compute GLVQ-like mu values for soft assignment
            same_class = (y[:, None] == proto_labels[None, :])
            diff_class = ~same_class
            INF = jnp.finfo(distances.dtype).max

            dp = jnp.min(jnp.where(same_class, distances, INF), axis=1)
            dm = jnp.min(jnp.where(diff_class, distances, INF), axis=1)
            mu = (dp - dm) / (dp + dm + 1e-10)

            # Weights: correctly classified samples get positive weight
            weights = jnp.where(mu < 0, -mu, mu * 0.1)  # focus on correct

            # Track loss
            loss = float(jnp.mean(mu))
            loss_history.append(loss)

            # M-step: for each prototype, find best replacement
            changed = False
            for k in range(prototypes.shape[0]):
                label_k = proto_labels[k]
                # Only consider data points with same label
                candidate_mask = (y == label_k)
                candidate_indices = jnp.where(candidate_mask, size=X.shape[0])[0]

                # Compute weighted distance sum for each candidate
                def eval_candidate(idx):
                    candidate = X[idx]
                    # Distance from all same-class samples to this candidate
                    d = jnp.sum((X - candidate[None, :]) ** 2, axis=1)
                    return jnp.sum(weights * d * candidate_mask)

                scores = jax.vmap(eval_candidate)(candidate_indices)
                # Mask out invalid (padded) candidates
                valid_mask = candidate_mask[candidate_indices]
                scores = jnp.where(valid_mask, scores, INF)

                best_idx = candidate_indices[jnp.argmin(scores)]
                new_proto = X[best_idx]

                if not jnp.allclose(prototypes[k], new_proto):
                    changed = True
                prototypes = prototypes.at[k].set(new_proto)

            if iteration > 0 and abs(loss_history[-1] - loss_history[-2]) < self.epsilon:
                break

        self.prototypes_ = prototypes
        self.prototype_labels_ = proto_labels
        self.loss_ = loss_history[-1]
        self.loss_history_ = jnp.array(loss_history)
        self.n_iter_ = iteration + 1
        return self
