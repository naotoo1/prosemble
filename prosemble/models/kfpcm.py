"""
JAX-based Kernel Fuzzy Possibilistic C-Means (KFPCM) clustering implementation.
"""

from typing import NamedTuple, Self
from functools import partial
import jax
import jax.numpy as jnp
import chex
from jax import jit
from prosemble.core.kernel import batch_gaussian_kernel
from prosemble.models.kfcm import KFCM
from prosemble.models.base import FuzzyClusteringBase, ScanFitMixin


class KFPCMState(NamedTuple):
    centroids: chex.Array
    U: chex.Array
    T: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class KFPCM(ScanFitMixin, FuzzyClusteringBase):
    """Kernel Fuzzy Possibilistic C-Means with JAX.

    KFPCM maintains two matrices (U and T) in kernel space. U has row-sum-to-1
    constraint (standard FCM), while T has column-sum-to-1 constraint per the
    original Pal, Pal & Bezdek (1997) FPCM formulation.
    """

    _hyperparams = ('fuzzifier', 'eta', 'sigma', 'init_method')
    _fitted_array_names = ('U_', 'T_')

    def __init__(self, n_clusters: int, fuzzifier: float = 2.0, eta: float = 2.0,
                 sigma: float = 1.0, max_iter: int = 100, epsilon: float = 1e-5,
                 init_method: str = 'kfcm', random_seed: int = 42, plot_steps: bool = False,
                 show_confidence: bool = True, show_pca_variance: bool = True,
                 save_plot_path: str | None = None, **kwargs):
        super().__init__(
            n_clusters=n_clusters, max_iter=max_iter, epsilon=epsilon,
            random_seed=random_seed, plot_steps=plot_steps,
            show_confidence=show_confidence, show_pca_variance=show_pca_variance,
            save_plot_path=save_plot_path, **kwargs
        )
        if fuzzifier <= 1.0:
            raise ValueError("fuzzifier must be > 1.0")
        if eta <= 1.0:
            raise ValueError("eta must be > 1.0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        self.fuzzifier = fuzzifier
        self.eta = eta
        self.sigma = sigma
        self.init_method = init_method

        # Model-specific fitted attributes
        self.U_ = None
        self.T_ = None

    def _initialize(self, X: chex.Array):
        n_samples = X.shape[0]
        if self.init_method == 'random':
            alpha = jnp.ones(self.n_clusters)
            U = jax.random.dirichlet(self.key, alpha, shape=(n_samples,))
            T = jax.random.dirichlet(self.key, alpha, shape=(n_samples,))
            indices = jax.random.choice(self.key, n_samples, shape=(self.n_clusters,), replace=False)
            centroids = X[indices]
        elif self.init_method == 'kfcm':
            kfcm = KFCM(self.n_clusters, self.fuzzifier, self.sigma, self.max_iter,
                           self.epsilon, 'random', self.random_seed, False)
            kfcm.fit(X)
            U = kfcm.U_
            centroids = kfcm.centroids_
            alpha = jnp.ones(self.n_clusters)
            T = jax.random.dirichlet(self.key, alpha, shape=(n_samples,))
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        return U, T, centroids

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(self, X: chex.Array, U: chex.Array, T: chex.Array, centroids: chex.Array) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.eta)
        weights = (U_fuzz + T_fuzz) * K
        numerator = weights.T @ X
        denominator = jnp.sum(weights, axis=0, keepdims=True).T
        return numerator / jnp.maximum(denominator, 1e-10)

    @partial(jit, static_argnums=(0,))
    def _update_fuzzy_matrix(self, X: chex.Array, centroids: chex.Array, fuzzifier: float) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 1.0 - K
        kernel_dist = jnp.maximum(kernel_dist, 1e-10)
        power = 1.0 / (fuzzifier - 1.0)
        def compute_membership_row(distances_i):
            ratios = distances_i[:, None] / distances_i[None, :]
            powered_ratios = jnp.power(ratios, power)
            denominators = jnp.sum(powered_ratios, axis=1)
            return 1.0 / denominators
        U = jax.vmap(compute_membership_row)(kernel_dist)
        return U / jnp.sum(U, axis=1, keepdims=True)

    @partial(jit, static_argnums=(0,))
    def _update_typicality_matrix(self, X: chex.Array, centroids: chex.Array) -> chex.Array:
        """Update typicality matrix with column-sum-to-1 constraint (Pal et al. 1997)."""
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 1.0 - K
        kernel_dist = jnp.maximum(kernel_dist, 1e-10)
        power = 1.0 / (self.eta - 1.0)
        inv_dist_powered = jnp.power(1.0 / kernel_dist, power)
        col_sums = jnp.maximum(jnp.sum(inv_dist_powered, axis=0, keepdims=True), 1e-10)
        return inv_dist_powered / col_sums

    @partial(jit, static_argnums=(0,))
    def _compute_objective(self, X: chex.Array, U: chex.Array, T: chex.Array, centroids: chex.Array) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.eta)
        weights = U_fuzz + T_fuzz
        return jnp.sum(weights * kernel_dist)

    @partial(jit, static_argnums=(0,))
    def _iteration_step(self, state: KFPCMState, X: chex.Array) -> tuple[KFPCMState, dict]:
        U_new = self._update_fuzzy_matrix(X, state.centroids, self.fuzzifier)
        T_new = self._update_typicality_matrix(X, state.centroids)
        centroids_new = self._compute_centroids(X, U_new, T_new, state.centroids)
        objective = self._compute_objective(X, U_new, T_new, centroids_new)
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon
        new_state = KFPCMState(centroids_new, U_new, T_new, objective, state.iteration + 1, converged)

        metrics = {
            'objective': new_state.objective,
            'centroid_change': centroid_change,
            'converged': new_state.converged,
        }

        return new_state, metrics

    def _build_info(self, state, iteration):
        labels = jnp.argmax(state.U, axis=1)
        weights = (jnp.max(state.U, axis=1) + jnp.max(state.T, axis=1)) / 2.0
        return {
            'centroids': state.centroids, 'labels': labels,
            'weights': weights, 'iteration': iteration,
            'objective': float(state.objective), 'max_iter': self.max_iter,
        }

    def fit(self, X: chex.Array, initial_centroids=None, resume=False) -> Self:
        if resume and initial_centroids is not None:
            raise ValueError("Cannot use both resume=True and initial_centroids")

        X = self._validate_input(X)

        if resume:
            self._check_fitted()
            centroids_init = self.centroids_
            U_init = self._update_fuzzy_matrix(X, centroids_init, self.fuzzifier)
            T_init = self._update_typicality_matrix(X, centroids_init)
        elif initial_centroids is not None:
            centroids_init = self._validate_initial_centroids(X, initial_centroids)
            U_init = self._update_fuzzy_matrix(X, centroids_init, self.fuzzifier)
            T_init = self._update_typicality_matrix(X, centroids_init)
        else:
            U_init, T_init, centroids_init = self._initialize(X)
        initial_objective = self._compute_objective(X, U_init, T_init, centroids_init)
        initial_state = KFPCMState(centroids_init, U_init, T_init, initial_objective, 0, False)

        final_state, self.history_ = self._run_training(X, initial_state)

        self.centroids_ = final_state.centroids
        self.U_ = final_state.U
        self.T_ = final_state.T
        self.n_iter_ = int(final_state.iteration)
        self.objective_ = float(final_state.objective)
        self.objective_history_ = self.history_['objective']
        return self

    def predict(self, X: chex.Array) -> chex.Array:
        self._check_fitted()
        U = self._update_fuzzy_matrix(X, self.centroids_, self.fuzzifier)
        return jnp.argmax(U, axis=1)

    def predict_proba(self, X: chex.Array) -> chex.Array:
        self._check_fitted()
        X = jnp.asarray(X)
        return self._update_fuzzy_matrix(X, self.centroids_, self.fuzzifier)

    def get_typicality(self, X: chex.Array) -> chex.Array:
        self._check_fitted()
        X = jnp.asarray(X)
        return self._update_typicality_matrix(X, self.centroids_)
