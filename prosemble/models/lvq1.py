"""
Learning Vector Quantization 1 (LVQ1).

Non-gradient, competitive learning algorithm. The winner prototype
is updated: attracted toward same-class samples, repelled from
different-class samples.

References
----------
.. [1] Kohonen, T. (1990). The Self-Organizing Map. Proc. IEEE.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel, NotFittedError
from prosemble.core.distance import squared_euclidean_distance_matrix
from prosemble.core.competitions import wtac


class LVQ1(SupervisedPrototypeModel):
    """Learning Vector Quantization 1.

    For each sample:
    - Find nearest prototype (winner)
    - If same class: w += lr * (x - w)  (attract)
    - If diff class: w -= lr * (x - w)  (repel)

    Uses batch updates (all samples per iteration).

    Parameters
    ----------
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
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
        """Fit LVQ1 using competitive learning (no gradients)."""
        X = jnp.asarray(X, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.int32)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        self.classes_ = jnp.unique(y)
        self.n_classes_ = int(len(self.classes_))

        # Initialize prototypes
        key = self.key
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key
        )

        if initial_prototypes is not None:
            prototypes = jnp.asarray(initial_prototypes, dtype=jnp.float32)

        loss_history = []

        for i in range(self.max_iter):
            # Compute distances and find winners
            distances = squared_euclidean_distance_matrix(X, prototypes)
            winners = jnp.argmin(distances, axis=1)  # (n,)

            # Determine if winner is same class
            winner_labels = proto_labels[winners]
            correct = (winner_labels == y)  # (n,)

            # Compute updates: attract if correct, repel if wrong
            signs = jnp.where(correct, 1.0, -1.0)  # (n,)
            diffs = X - prototypes[winners]  # (n, d)
            scaled_diffs = self.lr * signs[:, None] * diffs  # (n, d)

            # Accumulate updates per prototype
            n_protos = prototypes.shape[0]
            updates = jnp.zeros_like(prototypes)
            counts = jnp.zeros(n_protos)

            for k in range(n_protos):
                mask = (winners == k)
                if jnp.any(mask):
                    updates = updates.at[k].set(jnp.sum(scaled_diffs * mask[:, None], axis=0))
                    counts = counts.at[k].set(jnp.sum(mask))

            # Average update per prototype
            safe_counts = jnp.maximum(counts, 1.0)
            prototypes = prototypes + updates / safe_counts[:, None]

            # Track loss (mean distance to winner)
            winner_dists = distances[jnp.arange(X.shape[0]), winners]
            loss = float(jnp.mean(winner_dists))
            loss_history.append(loss)

            # Convergence check
            if i > 0 and abs(loss_history[-1] - loss_history[-2]) < self.epsilon:
                break

        self.prototypes_ = prototypes
        self.prototype_labels_ = proto_labels
        self.loss_ = loss_history[-1]
        self.loss_history_ = jnp.array(loss_history)
        self.n_iter_ = i + 1
        return self
