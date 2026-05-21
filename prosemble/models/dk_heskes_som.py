"""
Differentiating Kernel Heskes SOM (DKHeskesSOM).

Heskes SOM with Gaussian kernel distance:

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{\\|x - w_k\\|^2}{2\\sigma^2}
    \\right)\\right)

The kernel bandwidth :math:`\\sigma` is a fixed hyperparameter.
The Heskes BMU criterion and batch update remain unchanged.
Prototypes live in the original data space.

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

from prosemble.models.heskes_som import HeskesSOM, HeskesSOMState
from prosemble.core.kernel import kernel_distance_squared_per_proto


class DKHeskesSOM(HeskesSOM):
    """Differentiating Kernel Heskes SOM.

    Heskes SOM with Gaussian kernel distance. The kernel bandwidth
    :math:`\\sigma` is a fixed hyperparameter (not learned). The Heskes
    BMU criterion and batch update operate in the original data space —
    only the data-space distance metric changes.

    The Heskes BMU criterion uses kernel distance:

    .. math::

        c^*(x) = \\arg\\min_c \\sum_k h(k, c) \\cdot d_\\kappa^2(x, w_k)

    Parameters
    ----------
    kernel_sigma : float
        Gaussian kernel bandwidth for data-space distance. Default: 1.0.
    grid_height : int
        Height of the 2D grid.
    grid_width : int
        Width of the 2D grid.
    sigma_init : float, optional
        Initial grid neighborhood radius.
    sigma_final : float
        Final grid neighborhood radius.
    max_iter : int
        Maximum training iterations.
    epsilon : float
        Convergence threshold.
    random_seed : int
        Random seed.
    callbacks : list, optional
        Callback objects.
    use_scan : bool
        If True (default), use jax.lax.scan for training.
    patience : int, optional
        Epochs with no improvement before early stopping.
    restore_best : bool
        If True, restore best parameters after training.

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.

    See Also
    --------
    HeskesSOM : Base class with Euclidean distance.
    """

    def __init__(self, grid_height=10, grid_width=10, kernel_sigma=1.0,
                 sigma_init=None, sigma_final=0.5,
                 max_iter=100, lr=0.01, epsilon=1e-6, random_seed=42,
                 distance_fn=None, callbacks=None, use_scan=True,
                 patience=None, restore_best=False):
        super().__init__(
            grid_height=grid_height, grid_width=grid_width,
            sigma_init=sigma_init, sigma_final=sigma_final,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, distance_fn=distance_fn,
            callbacks=callbacks, use_scan=use_scan,
            patience=patience, restore_best=restore_best,
        )
        self.kernel_sigma = kernel_sigma

    def _kernel_distances(self, X, prototypes):
        """Compute kernel distances with broadcast sigma."""
        sigmas = jnp.full(prototypes.shape[0], self.kernel_sigma)
        return kernel_distance_squared_per_proto(X, prototypes, sigmas)

    @partial(jit, static_argnums=(0,))
    def _heskes_step(self, state, X, grid_dist_sq, sigma_init):
        """Single JIT-compiled DK Heskes SOM training step."""
        t = state.iteration
        max_t = jnp.array(max(self.max_iter - 1, 1), dtype=jnp.float32)
        frac = t.astype(jnp.float32) / max_t

        sigma_t = sigma_init * (self.sigma_final / sigma_init) ** frac

        prototypes = state.prototypes

        # Kernel distances: (n_samples, n_protos)
        distances = self._kernel_distances(X, prototypes)

        # Neighborhood matrix: h(k, c) for all pairs — (n_protos, n_protos)
        h_matrix = jnp.exp(-grid_dist_sq / (2.0 * sigma_t ** 2))

        # Heskes BMU: c*(x) = argmin_c Σ_k h(k,c) * d_κ(x,k)
        weighted_cost = jnp.dot(distances, h_matrix)  # (n_samples, n_protos)
        bmu_indices = jnp.argmin(weighted_cost, axis=1)

        # Neighborhood weights for each sample: h(k, c*(x))
        h_weights = h_matrix[:, bmu_indices].T  # (n_samples, n_protos)

        # Batch update: w_k = Σ_x h(k,c*(x)) * x / Σ_x h(k,c*(x))
        numerator = jnp.dot(h_weights.T, X)  # (n_protos, n_features)
        denominator = jnp.sum(h_weights, axis=0)[:, None]
        new_prototypes = numerator / (denominator + 1e-10)

        # Heskes energy: E = Σ_x Σ_k h(k,c*(x)) * d_κ(x,k)
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

    def _fit_with_python_loop(self, X, prototypes, grid_dist_sq,
                               sigma_init_val):
        """Python for-loop training with kernel distance."""
        loss_history = []
        best_loss = None
        best_prototypes = None

        for t in range(self.max_iter):
            frac = t / max(self.max_iter - 1, 1)
            sigma_t = sigma_init_val * (
                self.sigma_final / sigma_init_val
            ) ** frac

            distances = self._kernel_distances(X, prototypes)
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
        """Return BMU grid coordinates using Heskes criterion with kernel distance.

        Parameters
        ----------
        X : array of shape (n, d)

        Returns
        -------
        coords : array of shape (n, 2) — (row, col) for each sample
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        distances = self._kernel_distances(X, self.prototypes_)

        grid_pos = self._grid_positions
        grid_dist_sq = jnp.sum(
            (grid_pos[:, None, :] - grid_pos[None, :, :]) ** 2, axis=2
        )
        h_matrix = jnp.exp(-grid_dist_sq / (2.0 * self.sigma_final ** 2))
        weighted_cost = jnp.dot(distances, h_matrix)
        bmu_indices = jnp.argmin(weighted_cost, axis=1)
        return self._grid_positions[bmu_indices]

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['kernel_sigma'] = self.kernel_sigma
        return hp
