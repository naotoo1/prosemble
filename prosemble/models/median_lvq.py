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
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    random_seed : int
        Random seed.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('optimizer', 'adam')
        super().__init__(**kwargs)

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
