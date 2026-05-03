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
    - Find closest same-class prototype w+ and closest different-class w-
    - w+ += lr * (x - w+)   (attract w+)
    - w- -= lr * (x - w-)   (repel w-)

    Parameters
    ----------
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('optimizer', 'adam')
        super().__init__(**kwargs)

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
