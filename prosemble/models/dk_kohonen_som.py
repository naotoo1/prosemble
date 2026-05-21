"""
Differentiating Kernel Kohonen SOM (DKKohonenSOM).

Standard Kohonen SOM with Gaussian kernel distance for BMU selection:

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{\\|x - w_k\\|^2}{2\\sigma^2}
    \\right)\\right)

The kernel bandwidth :math:`\\sigma` is a fixed hyperparameter.
Grid neighborhood is unchanged. Prototypes live in the original data space.

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

from prosemble.models.kohonen_som import KohonenSOM, SOMState
from prosemble.core.kernel import kernel_distance_squared_per_proto


class DKKohonenSOM(KohonenSOM):
    """Differentiating Kernel Kohonen SOM.

    Standard Kohonen SOM with Gaussian kernel distance for BMU selection.
    The kernel bandwidth :math:`\\sigma` is a fixed hyperparameter (not learned).
    The grid-based neighborhood and competitive update rule operate in the
    original data space — only the data-space distance metric changes.

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
    lr_init : float
        Initial learning rate.
    lr_final : float
        Final learning rate.
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
    KohonenSOM : Base class with Euclidean distance.
    """

    def __init__(self, grid_height=10, grid_width=10, kernel_sigma=1.0,
                 sigma_init=None, sigma_final=0.5,
                 lr_init=0.5, lr_final=0.01,
                 max_iter=100, lr=0.01, epsilon=1e-6, random_seed=42,
                 distance_fn=None, callbacks=None, use_scan=True,
                 patience=None, restore_best=False):
        super().__init__(
            grid_height=grid_height, grid_width=grid_width,
            sigma_init=sigma_init, sigma_final=sigma_final,
            lr_init=lr_init, lr_final=lr_final,
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
    def _som_step(self, state, X, grid_dist_sq, sigma_init):
        """Single JIT-compiled DK Kohonen SOM training step."""
        t = state.iteration
        max_t = jnp.array(max(self.max_iter - 1, 1), dtype=jnp.float32)
        frac = t.astype(jnp.float32) / max_t

        sigma_t = sigma_init * (self.sigma_final / sigma_init) ** frac
        lr_t = self.lr_init * (self.lr_final / self.lr_init) ** frac

        prototypes = state.prototypes
        n_samples = X.shape[0]

        # BMU via kernel distance
        distances = self._kernel_distances(X, prototypes)
        bmu_indices = jnp.argmin(distances, axis=1)

        # Gaussian neighborhood in grid space
        bmu_grid_dist_sq = grid_dist_sq[bmu_indices]
        h = jnp.exp(-bmu_grid_dist_sq / (2.0 * sigma_t ** 2))

        # Batch update in data space
        diffs = X[:, None, :] - prototypes[None, :, :]
        weighted_diffs = h[:, :, None] * diffs
        numerator = jnp.sum(weighted_diffs, axis=0)
        denominator = jnp.sum(h, axis=0)[:, None]
        update = lr_t * numerator / (denominator + 1e-10)

        new_prototypes = prototypes + update

        # Quantization error (kernel distance)
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

    def _fit_with_python_loop(self, X, prototypes, grid_dist_sq, sigma_init_val):
        """Python for-loop training with kernel distance."""
        n_samples = X.shape[0]
        loss_history = []
        best_loss = None
        best_prototypes = None

        for t in range(self.max_iter):
            frac = t / max(self.max_iter - 1, 1)
            sigma_t = sigma_init_val * (self.sigma_final / sigma_init_val) ** frac
            lr_t = self.lr_init * (self.lr_final / self.lr_init) ** frac

            distances = self._kernel_distances(X, prototypes)
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
        """Return BMU grid coordinates using kernel distance.

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
        bmu_indices = jnp.argmin(distances, axis=1)
        return self._grid_positions[bmu_indices]

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['kernel_sigma'] = self.kernel_sigma
        return hp
