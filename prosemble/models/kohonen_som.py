"""
Kohonen Self-Organizing Map (standard textbook algorithm).

This implements the standard Kohonen SOM with Gaussian neighborhood
and exponential decay, distinct from prosemble's existing SOM.

References
----------
.. [1] Kohonen, T. (1990). The Self-Organizing Map. Proc. IEEE.
"""

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax

from prosemble.models.prototype_base import UnsupervisedPrototypeModel
from prosemble.core.distance import squared_euclidean_distance_matrix


class SOMState(NamedTuple):
    """State for Kohonen SOM lax.scan loop."""
    prototypes: jnp.ndarray
    loss: jnp.ndarray
    prev_loss: jnp.ndarray
    converged: jnp.ndarray
    iteration: jnp.ndarray


class KohonenSOM(UnsupervisedPrototypeModel):
    """Standard Kohonen Self-Organizing Map.

    Uses squared Euclidean distance for BMU selection, Gaussian
    neighborhood function, exponential decay for sigma and learning
    rate, and batch updates.

    Parameters
    ----------
    grid_height : int
        Height of the 2D grid.
    grid_width : int
        Width of the 2D grid.
    sigma_init : float
        Initial neighborhood radius.
    sigma_final : float
        Final neighborhood radius.
    lr_init : float
        Initial learning rate.
    lr_final : float
        Final learning rate.
    max_iter : int
        Number of training epochs.

    See Also
    --------
    UnsupervisedPrototypeModel : Full list of base parameters (distance_fn,
        callbacks, use_scan, patience, etc.).
    """

    def __init__(self, grid_height=10, grid_width=10,
                 sigma_init=None, sigma_final=0.5,
                 lr_init=0.5, lr_final=0.01, **kwargs):
        n_prototypes = grid_height * grid_width
        kwargs['n_prototypes'] = n_prototypes
        super().__init__(**kwargs)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.sigma_init = sigma_init  # default: max(h, w) / 2
        self.sigma_final = sigma_final
        self.lr_init = lr_init
        self.lr_final = lr_final

        # Precompute grid positions
        rows, cols = jnp.meshgrid(
            jnp.arange(grid_height), jnp.arange(grid_width), indexing='ij'
        )
        self._grid_positions = jnp.stack([rows.ravel(), cols.ravel()], axis=1).astype(jnp.float32)

    @partial(jit, static_argnums=(0,))
    def _som_step(self, state, X, grid_dist_sq, sigma_init):
        """Single JIT-compiled Kohonen SOM training step."""
        t = state.iteration
        max_t = jnp.array(max(self.max_iter - 1, 1), dtype=jnp.float32)
        frac = t.astype(jnp.float32) / max_t

        sigma_t = sigma_init * (self.sigma_final / sigma_init) ** frac
        lr_t = self.lr_init * (self.lr_final / self.lr_init) ** frac

        prototypes = state.prototypes
        n_samples = X.shape[0]

        # Find BMU
        distances = squared_euclidean_distance_matrix(X, prototypes)
        bmu_indices = jnp.argmin(distances, axis=1)

        # Gaussian neighborhood
        bmu_grid_dist_sq = grid_dist_sq[bmu_indices]
        h = jnp.exp(-bmu_grid_dist_sq / (2.0 * sigma_t ** 2))

        # Batch update
        diffs = X[:, None, :] - prototypes[None, :, :]
        weighted_diffs = h[:, :, None] * diffs
        numerator = jnp.sum(weighted_diffs, axis=0)
        denominator = jnp.sum(h, axis=0)[:, None]
        update = lr_t * numerator / (denominator + 1e-10)

        new_prototypes = prototypes + update

        # Quantization error
        bmu_dists = distances[jnp.arange(n_samples), bmu_indices]
        qe = jnp.mean(bmu_dists)

        # Convergence
        has_converged = state.converged | (
            jnp.abs(qe - state.prev_loss) < self.epsilon
        )
        frozen_prototypes = jnp.where(state.converged, prototypes, new_prototypes)
        frozen_qe = jnp.where(state.converged, state.loss, qe)

        new_state = SOMState(
            prototypes=frozen_prototypes,
            loss=frozen_qe,
            prev_loss=qe,
            converged=has_converged,
            iteration=t + 1,
        )
        return new_state, frozen_qe

    @partial(jit, static_argnums=(0,))
    def _fit_scan(self, X, prototypes, grid_dist_sq, sigma_init):
        """Scan-based training loop."""
        initial_state = SOMState(
            prototypes=prototypes,
            loss=jnp.array(float('inf')),
            prev_loss=jnp.array(float('inf')),
            converged=jnp.array(False),
            iteration=jnp.array(0),
        )

        def scan_fn(state, _):
            return self._som_step(state, X, grid_dist_sq, sigma_init)

        final_state, loss_history = lax.scan(
            scan_fn, initial_state, None, length=self.max_iter
        )
        return final_state, loss_history

    def fit(self, X):
        """Fit KohonenSOM."""
        X = jnp.asarray(X, dtype=jnp.float32)
        n_samples = X.shape[0]

        key = self.key
        indices = jax.random.choice(key, n_samples, (self.n_prototypes,), replace=False)
        prototypes = X[indices]

        sigma_init_val = self.sigma_init if self.sigma_init else max(self.grid_height, self.grid_width) / 2.0

        # Precompute grid distances
        grid_pos = self._grid_positions
        grid_dist_sq = jnp.sum(
            (grid_pos[:, None, :] - grid_pos[None, :, :]) ** 2, axis=2
        )

        if self.use_scan and self.patience is None and not self.restore_best:
            return self._fit_with_scan(X, prototypes, grid_dist_sq, sigma_init_val)
        else:
            return self._fit_with_python_loop(X, prototypes, grid_dist_sq, sigma_init_val)

    def _fit_with_scan(self, X, prototypes, grid_dist_sq, sigma_init_val):
        """lax.scan training: JIT-compiled, runs all max_iter iterations."""
        sigma_init = jnp.array(sigma_init_val, dtype=jnp.float32)
        final_state, loss_history = self._fit_scan(X, prototypes, grid_dist_sq, sigma_init)

        converged_mask = jnp.abs(jnp.diff(loss_history)) < self.epsilon
        first_converged = jnp.argmax(converged_mask)
        has_any = jnp.any(converged_mask)
        n_iter = jnp.where(has_any, first_converged + 2, self.max_iter)

        self.prototypes_ = final_state.prototypes
        self.n_iter_ = int(n_iter)
        self.loss_ = float(final_state.loss)
        self.loss_history_ = loss_history
        return self

    def _fit_with_python_loop(self, X, prototypes, grid_dist_sq, sigma_init_val):
        """Python for-loop training: true early stopping, no wasted compute."""
        n_samples = X.shape[0]
        loss_history = []
        best_loss = None
        best_prototypes = None

        for t in range(self.max_iter):
            frac = t / max(self.max_iter - 1, 1)
            sigma_t = sigma_init_val * (self.sigma_final / sigma_init_val) ** frac
            lr_t = self.lr_init * (self.lr_final / self.lr_init) ** frac

            distances = squared_euclidean_distance_matrix(X, prototypes)
            bmu_indices = jnp.argmin(distances, axis=1)

            bmu_grid_dist_sq = grid_dist_sq[bmu_indices]
            h = jnp.exp(-bmu_grid_dist_sq / (2.0 * sigma_t ** 2))

            diffs = X[:, None, :] - prototypes[None, :, :]
            weighted_diffs = h[:, :, None] * diffs
            numerator = jnp.sum(weighted_diffs, axis=0)
            denominator = jnp.sum(h, axis=0)[:, None]
            update = lr_t * numerator / (denominator + 1e-10)
            prototypes = prototypes + update

            bmu_dists = distances[jnp.arange(n_samples), bmu_indices]
            qe = float(jnp.mean(bmu_dists))
            loss_history.append(qe)

            if self.restore_best and (best_loss is None or qe < best_loss):
                best_loss = qe
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
        """Return BMU grid coordinates for each sample.

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
        bmu_indices = jnp.argmin(distances, axis=1)
        return self._grid_positions[bmu_indices]

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'sigma_final': self.sigma_final,
            'lr_init': self.lr_init,
            'lr_final': self.lr_final,
        })
        if self.sigma_init is not None:
            hp['sigma_init'] = self.sigma_init
        return hp
