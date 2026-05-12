"""
Heskes Self-Organizing Map.

Unlike the standard Kohonen SOM, the Heskes SOM uses a modified
Best Matching Unit (BMU) definition that accounts for the neighborhood
function, and a pure batch update rule (no learning rate). This
guarantees monotonic decrease of a well-defined energy function.

Energy function::

    E = Σ_x Σ_k h(k, c*(x)) * ||x - w_k||^2

Modified BMU::

    c*(x) = argmin_c Σ_k h(k, c) * ||x - w_k||^2

Batch update::

    w_k = Σ_x h(k, c*(x)) * x / Σ_x h(k, c*(x))

References
----------
.. [1] Heskes, T. (1999). Energy functions for self-organizing maps.
       In Kohonen Maps, pp. 303-316, Elsevier.
.. [2] Heskes, T. (2001). Self-organizing maps, vector quantization,
       and mixture modeling. IEEE Trans. Neural Networks, 12(6).
"""

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax

from prosemble.models.prototype_base import UnsupervisedPrototypeModel
from prosemble.core.distance import squared_euclidean_distance_matrix


class HeskesSOMState(NamedTuple):
    """State for Heskes SOM lax.scan loop."""
    prototypes: jnp.ndarray
    loss: jnp.ndarray
    prev_loss: jnp.ndarray
    converged: jnp.ndarray
    iteration: jnp.ndarray


class HeskesSOM(UnsupervisedPrototypeModel):
    """Heskes Self-Organizing Map.

    Uses a modified BMU definition that considers the neighborhood
    structure, and a pure batch update (weighted average of data).
    Guarantees monotonic decrease of the Heskes energy function.

    Differences from KohonenSOM:

    - BMU is chosen to minimize neighborhood-weighted distance sum,
      not raw distance to closest prototype.
    - Prototypes are updated via weighted average (no learning rate).
    - Energy is guaranteed to decrease monotonically.

    Parameters
    ----------
    grid_height : int
        Height of the 2D grid.
    grid_width : int
        Width of the 2D grid.
    sigma_init : float, optional
        Initial neighborhood radius. Default: max(h, w) / 2.
    sigma_final : float
        Final neighborhood radius.
    max_iter : int
        Number of training epochs.

    See Also
    --------
    UnsupervisedPrototypeModel : Full list of base parameters (distance_fn,
        callbacks, use_scan, patience, etc.).
    """

    def __init__(self, grid_height=10, grid_width=10,
                 sigma_init=None, sigma_final=0.5, **kwargs):
        n_prototypes = grid_height * grid_width
        kwargs['n_prototypes'] = n_prototypes
        super().__init__(**kwargs)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final

        # Precompute grid positions
        rows, cols = jnp.meshgrid(
            jnp.arange(grid_height), jnp.arange(grid_width), indexing='ij'
        )
        self._grid_positions = jnp.stack(
            [rows.ravel(), cols.ravel()], axis=1
        ).astype(jnp.float32)

    @partial(jit, static_argnums=(0,))
    def _heskes_step(self, state, X, grid_dist_sq, sigma_init):
        """Single JIT-compiled Heskes SOM training step."""
        t = state.iteration
        max_t = jnp.array(max(self.max_iter - 1, 1), dtype=jnp.float32)
        frac = t.astype(jnp.float32) / max_t

        sigma_t = sigma_init * (self.sigma_final / sigma_init) ** frac

        prototypes = state.prototypes
        n_protos = prototypes.shape[0]
        n_samples = X.shape[0]

        # Squared distances: (n_samples, n_protos)
        distances = squared_euclidean_distance_matrix(X, prototypes)

        # Neighborhood matrix: h(k, c) for all pairs — (n_protos, n_protos)
        h_matrix = jnp.exp(-grid_dist_sq / (2.0 * sigma_t ** 2))

        # Heskes BMU: c*(x) = argmin_c Σ_k h(k,c) * ||x - w_k||^2
        # For each sample x, compute Σ_k h(k,c) * d(x,k) for each candidate c
        # distances: (n_samples, n_protos=k), h_matrix: (k, c)
        # weighted_cost: (n_samples, c)
        weighted_cost = jnp.dot(distances, h_matrix)  # (n_samples, n_protos)
        bmu_indices = jnp.argmin(weighted_cost, axis=1)  # (n_samples,)

        # Neighborhood weights for each sample: h(k, c*(x))
        # bmu_indices: (n_samples,) -> h_matrix[bmu_indices]: would be wrong axis
        # h_matrix is (k, c), we need h(k, c*(x)) for each x
        # h_matrix[:, bmu_indices] -> (n_protos=k, n_samples) -> transpose
        h_weights = h_matrix[:, bmu_indices].T  # (n_samples, n_protos=k)

        # Batch update: w_k = Σ_x h(k, c*(x)) * x / Σ_x h(k, c*(x))
        # h_weights: (n_samples, k), X: (n_samples, d)
        # numerator: (k, d) = h_weights.T @ X
        numerator = jnp.dot(h_weights.T, X)  # (n_protos, n_features)
        denominator = jnp.sum(h_weights, axis=0)[:, None]  # (n_protos, 1)
        new_prototypes = numerator / (denominator + 1e-10)

        # Heskes energy: E = Σ_x Σ_k h(k, c*(x)) * ||x - w_k||^2
        energy = jnp.sum(h_weights * distances)

        # Convergence
        has_converged = state.converged | (
            jnp.abs(energy - state.prev_loss) < self.epsilon
        )
        frozen_prototypes = jnp.where(
            state.converged, prototypes, new_prototypes
        )
        frozen_energy = jnp.where(state.converged, state.loss, energy)

        new_state = HeskesSOMState(
            prototypes=frozen_prototypes,
            loss=frozen_energy,
            prev_loss=energy,
            converged=has_converged,
            iteration=t + 1,
        )
        return new_state, frozen_energy

    @partial(jit, static_argnums=(0,))
    def _fit_scan(self, X, prototypes, grid_dist_sq, sigma_init):
        """Scan-based training loop."""
        initial_state = HeskesSOMState(
            prototypes=prototypes,
            loss=jnp.array(float('inf')),
            prev_loss=jnp.array(float('inf')),
            converged=jnp.array(False),
            iteration=jnp.array(0),
        )

        def scan_fn(state, _):
            return self._heskes_step(state, X, grid_dist_sq, sigma_init)

        final_state, loss_history = lax.scan(
            scan_fn, initial_state, None, length=self.max_iter
        )
        return final_state, loss_history

    def fit(self, X):
        """Fit HeskesSOM."""
        X = jnp.asarray(X, dtype=jnp.float32)
        n_samples = X.shape[0]

        key = self.key
        indices = jax.random.choice(
            key, n_samples, (self.n_prototypes,), replace=False
        )
        prototypes = X[indices]

        sigma_init_val = (
            self.sigma_init if self.sigma_init
            else max(self.grid_height, self.grid_width) / 2.0
        )

        # Precompute grid distances
        grid_pos = self._grid_positions
        grid_dist_sq = jnp.sum(
            (grid_pos[:, None, :] - grid_pos[None, :, :]) ** 2, axis=2
        )

        if self.use_scan and self.patience is None and not self.restore_best:
            return self._fit_with_scan(
                X, prototypes, grid_dist_sq, sigma_init_val
            )
        else:
            return self._fit_with_python_loop(
                X, prototypes, grid_dist_sq, sigma_init_val
            )

    def _fit_with_scan(self, X, prototypes, grid_dist_sq, sigma_init_val):
        """lax.scan training."""
        sigma_init = jnp.array(sigma_init_val, dtype=jnp.float32)
        final_state, loss_history = self._fit_scan(
            X, prototypes, grid_dist_sq, sigma_init
        )

        converged_mask = jnp.abs(jnp.diff(loss_history)) < self.epsilon
        first_converged = jnp.argmax(converged_mask)
        has_any = jnp.any(converged_mask)
        n_iter = jnp.where(has_any, first_converged + 2, self.max_iter)

        self.prototypes_ = final_state.prototypes
        self.n_iter_ = int(n_iter)
        self.loss_ = float(final_state.loss)
        self.loss_history_ = loss_history
        return self

    def _fit_with_python_loop(self, X, prototypes, grid_dist_sq,
                               sigma_init_val):
        """Python for-loop training with true early stopping."""
        n_samples = X.shape[0]
        loss_history = []
        best_loss = None
        best_prototypes = None

        # Neighborhood matrix: h(k, c)
        for t in range(self.max_iter):
            frac = t / max(self.max_iter - 1, 1)
            sigma_t = sigma_init_val * (
                self.sigma_final / sigma_init_val
            ) ** frac

            distances = squared_euclidean_distance_matrix(X, prototypes)
            h_matrix = jnp.exp(-grid_dist_sq / (2.0 * sigma_t ** 2))

            # Heskes BMU
            weighted_cost = jnp.dot(distances, h_matrix)
            bmu_indices = jnp.argmin(weighted_cost, axis=1)

            # Neighborhood weights
            h_weights = h_matrix[:, bmu_indices].T

            # Batch update
            numerator = jnp.dot(h_weights.T, X)
            denominator = jnp.sum(h_weights, axis=0)[:, None]
            prototypes = numerator / (denominator + 1e-10)

            # Energy
            energy = float(jnp.sum(h_weights * distances))
            loss_history.append(energy)

            if self.restore_best and (best_loss is None or energy < best_loss):
                best_loss = energy
                best_prototypes = prototypes

            if t > 0 and abs(loss_history[-1] - loss_history[-2]) < self.epsilon:
                break
            if self.patience is not None and self._check_patience(loss_history, self.patience):
                break

        if self.restore_best and best_prototypes is not None:
            prototypes = best_prototypes
            self.best_loss_ = best_loss

        self.prototypes_ = prototypes
        self.n_iter_ = t + 1
        self.loss_ = loss_history[-1]
        self.loss_history_ = jnp.array(loss_history)
        return self

    def bmu_map(self, X):
        """Return BMU grid coordinates using Heskes criterion.

        Parameters
        ----------
        X : array of shape (n, d)

        Returns
        -------
        coords : array of shape (n, 2) — (row, col) for each sample
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        distances = squared_euclidean_distance_matrix(X, self.prototypes_)

        grid_pos = self._grid_positions
        grid_dist_sq = jnp.sum(
            (grid_pos[:, None, :] - grid_pos[None, :, :]) ** 2, axis=2
        )
        # Use a small sigma for inference (tight neighborhood)
        h_matrix = jnp.exp(-grid_dist_sq / (2.0 * self.sigma_final ** 2))
        weighted_cost = jnp.dot(distances, h_matrix)
        bmu_indices = jnp.argmin(weighted_cost, axis=1)
        return self._grid_positions[bmu_indices]

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'sigma_final': self.sigma_final,
        })
        if self.sigma_init is not None:
            hp['sigma_init'] = self.sigma_init
        return hp
