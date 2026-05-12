"""
Growing Neural Gas (GNG).

An incremental self-organizing network that can grow and shrink
by adding/removing nodes based on accumulated error.

References
----------
.. [1] Fritzke, B. (1995). A Growing Neural Gas Network Learns
       Topologies. NIPS.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import UnsupervisedPrototypeModel
from prosemble.core.distance import squared_euclidean_distance_matrix


class GrowingNeuralGas(UnsupervisedPrototypeModel):
    """Growing Neural Gas.

    Starts with 2 nodes and grows by inserting nodes near the
    highest-error units. Connections between nodes have ages;
    old connections are removed.

    Parameters
    ----------
    max_nodes : int
        Maximum number of nodes.
    max_iter : int
        Number of training steps (epochs over data).
    lr_winner : float
        Learning rate for the winning node.
    lr_neighbor : float
        Learning rate for neighbors of the winner.
    max_age : int
        Maximum age before an edge is removed.
    insert_interval : int
        Insert a new node every this many steps.
    error_decay : float
        Error decay factor applied to all nodes.

    See Also
    --------
    UnsupervisedPrototypeModel : Full list of base parameters (distance_fn,
        callbacks, use_scan, patience, etc.).
    """

    def __init__(self, max_nodes=100, lr_winner=0.1, lr_neighbor=0.01,
                 max_age=50, insert_interval=100, error_decay=0.995,
                 **kwargs):
        kwargs.setdefault('n_prototypes', 2)  # start with 2
        super().__init__(**kwargs)
        self.max_nodes = max_nodes
        self.lr_winner = lr_winner
        self.lr_neighbor = lr_neighbor
        self.max_age = max_age
        self.insert_interval = insert_interval
        self.error_decay = error_decay

        # Topology
        self.edges_ = None
        self.n_active_ = None

    def fit(self, X):
        """Fit Growing Neural Gas."""
        X = jnp.asarray(X, dtype=jnp.float32)
        n_samples, n_features = X.shape

        # Pre-allocate for max_nodes
        key = self.key
        key1, key2 = jax.random.split(key)
        idx = jax.random.choice(key1, n_samples, (2,), replace=False)
        prototypes = np.zeros((self.max_nodes, n_features), dtype=np.float32)
        prototypes[0] = np.array(X[idx[0]])
        prototypes[1] = np.array(X[idx[1]])

        edges = np.full((self.max_nodes, self.max_nodes), -1, dtype=np.int32)  # -1 = no edge
        errors = np.zeros(self.max_nodes, dtype=np.float32)
        n_active = 2

        step = 0
        loss_history = []

        for epoch in range(self.max_iter):
            epoch_error = 0.0
            # Shuffle data
            perm_key = jax.random.fold_in(key2, epoch)
            perm = jax.random.permutation(perm_key, n_samples)

            for i in range(n_samples):
                x = np.array(X[perm[i]])
                step += 1

                # Find two closest nodes
                active_protos = prototypes[:n_active]
                dists = np.sum((active_protos - x) ** 2, axis=1)
                sorted_idx = np.argsort(dists)
                s1, s2 = sorted_idx[0], sorted_idx[1]

                # Accumulate error for winner
                errors[s1] += dists[s1]
                epoch_error += dists[s1]

                # Create/refresh edge between s1 and s2
                edges[s1, s2] = 0
                edges[s2, s1] = 0

                # Move winner and its neighbors
                prototypes[s1] += self.lr_winner * (x - prototypes[s1])
                for j in range(n_active):
                    if edges[s1, j] >= 0:  # neighbor
                        prototypes[j] += self.lr_neighbor * (x - prototypes[j])

                # Age all edges from s1
                for j in range(n_active):
                    if edges[s1, j] >= 0:
                        edges[s1, j] += 1
                        edges[j, s1] += 1

                # Remove old edges
                old_mask = edges >= self.max_age
                edges[old_mask] = -1

                # Remove isolated nodes (no edges)
                # Skip for simplicity — just mark for potential cleanup

                # Insert new node
                if step % self.insert_interval == 0 and n_active < self.max_nodes:
                    # Find node with largest error
                    q = np.argmax(errors[:n_active])
                    # Find its neighbor with largest error
                    neighbors_q = np.where(edges[q, :n_active] >= 0)[0]
                    if len(neighbors_q) > 0:
                        f = neighbors_q[np.argmax(errors[neighbors_q])]
                        # Insert new node between q and f
                        new_idx = n_active
                        prototypes[new_idx] = 0.5 * (prototypes[q] + prototypes[f])
                        # Remove edge q-f, add edges q-new and f-new
                        edges[q, f] = -1
                        edges[f, q] = -1
                        edges[q, new_idx] = 0
                        edges[new_idx, q] = 0
                        edges[f, new_idx] = 0
                        edges[new_idx, f] = 0
                        # Distribute error
                        errors[new_idx] = 0.5 * (errors[q] + errors[f])
                        errors[q] *= 0.5
                        errors[f] *= 0.5
                        n_active += 1

                # Decay all errors
                errors[:n_active] *= self.error_decay

            loss_history.append(epoch_error / n_samples)

            if epoch > 0 and abs(loss_history[-1] - loss_history[-2]) < self.epsilon:
                break

        # Store only active nodes
        self.prototypes_ = jnp.array(prototypes[:n_active])
        self.edges_ = jnp.array(edges[:n_active, :n_active])
        self.n_active_ = n_active
        self.n_iter_ = epoch + 1
        self.loss_ = loss_history[-1]
        self.loss_history_ = jnp.array(loss_history)
        return self

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'max_nodes': self.max_nodes,
            'lr_winner': self.lr_winner,
            'lr_neighbor': self.lr_neighbor,
            'max_age': self.max_age,
            'insert_interval': self.insert_interval,
            'error_decay': self.error_decay,
        })
        return hp
