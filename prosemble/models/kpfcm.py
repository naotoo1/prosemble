"""
JAX-based Kernel Possibilistic Fuzzy C-Means (KPFCM) clustering implementation.
"""

from typing import NamedTuple, Self
from functools import partial
import jax
import jax.numpy as jnp
import chex
from jax import jit
from prosemble.core.kernel import batch_gaussian_kernel
from prosemble.models.base import FuzzyClusteringBase, ScanFitMixin
from prosemble.models.kfcm import KFCM


class KPFCMState(NamedTuple):
    centroids: chex.Array
    U: chex.Array
    T: chex.Array
    gamma: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class KPFCM(ScanFitMixin, FuzzyClusteringBase):
    """Kernel Possibilistic Fuzzy C-Means with JAX.

    KPFCM combines fuzzy membership (U) and typicality (T) in kernel space
    with weights a and b.

    See Also
    --------
    FuzzyClusteringBase : Full list of base parameters (distance_fn,
        patience, restore_best, callbacks, etc.).
    """

    _hyperparams = ('fuzzifier', 'sigma', 'a', 'b', 'eta', 'k', 'init_method')
    _fitted_array_names = ('U_', 'T_', 'gamma_')

    def __init__(self, n_clusters: int, fuzzifier: float = 2.0, eta: float = 2.0,
                 a: float = 1.0, b: float = 1.0, k: float = 1.0, sigma: float = 1.0,
                 max_iter: int = 100, epsilon: float = 1e-5, init_method: str = 'kfcm',
                 random_seed: int = 42, plot_steps: bool = False, show_confidence: bool = True,
                 show_pca_variance: bool = True, save_plot_path: str | None = None, **kwargs):
        # Validate model-specific parameters
        if fuzzifier <= 1.0:
            raise ValueError("fuzzifier must be > 1.0")
        if eta <= 1.0:
            raise ValueError("eta must be > 1.0")
        if a <= 0:
            raise ValueError("a must be > 0")
        if b <= 0:
            raise ValueError("b must be > 0")
        if k <= 0:
            raise ValueError("k must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            epsilon=epsilon,
            random_seed=random_seed,
            plot_steps=plot_steps,
            show_confidence=show_confidence,
            show_pca_variance=show_pca_variance,
            save_plot_path=save_plot_path, **kwargs
        )

        self.fuzzifier = fuzzifier
        self.eta = eta
        self.a = a
        self.b = b
        self.k = k
        self.sigma = sigma
        self.init_method = init_method

        # Model-specific fitted attributes
        self.U_ = None
        self.T_ = None
        self.gamma_ = None

    def _initialize(self, X: chex.Array):
        kfcm = KFCM(self.n_clusters, self.fuzzifier, self.sigma, self.max_iter,
                       self.epsilon, 'random', self.random_seed, False)
        kfcm.fit(X)
        U = kfcm.U_
        centroids = kfcm.centroids_
        T = jnp.ones_like(U) / self.n_clusters
        return U, T, centroids

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(self, X: chex.Array, U: chex.Array, T: chex.Array, centroids: chex.Array) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.eta)
        weights = (self.a * U_fuzz + self.b * T_fuzz) * K
        numerator = weights.T @ X
        denominator = jnp.sum(weights, axis=0, keepdims=True).T
        return numerator / jnp.maximum(denominator, 1e-10)

    @partial(jit, static_argnums=(0,))
    def _update_U(self, X: chex.Array, centroids: chex.Array) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 1.0 - K
        kernel_dist = jnp.maximum(kernel_dist, 1e-10)
        power = 1.0 / (self.fuzzifier - 1.0)
        def compute_membership_row(distances_i):
            ratios = distances_i[:, None] / distances_i[None, :]
            powered_ratios = jnp.power(ratios, power)
            denominators = jnp.sum(powered_ratios, axis=1)
            return 1.0 / denominators
        U = jax.vmap(compute_membership_row)(kernel_dist)
        return U / jnp.sum(U, axis=1, keepdims=True)

    @partial(jit, static_argnums=(0,))
    def _compute_gamma(self, X: chex.Array, U: chex.Array, centroids: chex.Array) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        U_fuzz = jnp.power(U, self.fuzzifier)
        numerator = jnp.sum(U_fuzz * kernel_dist, axis=0)
        denominator = jnp.sum(U_fuzz, axis=0)
        return self.k * numerator / denominator

    @partial(jit, static_argnums=(0,))
    def _update_T(self, X: chex.Array, centroids: chex.Array, gamma: chex.Array) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        kernel_dist = jnp.maximum(kernel_dist, 1e-10)
        power = 1.0 / (self.eta - 1.0)
        ratio = self.b * kernel_dist / gamma[None, :]
        return 1.0 / (1.0 + jnp.power(ratio, power))

    @partial(jit, static_argnums=(0,))
    def _compute_objective(self, X: chex.Array, U: chex.Array, T: chex.Array,
                          centroids: chex.Array, gamma: chex.Array) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.eta)
        weights = self.a * U_fuzz + self.b * T_fuzz
        term1 = jnp.sum(weights * kernel_dist)
        one_minus_T = 1.0 - T
        one_minus_T_fuzz = jnp.power(one_minus_T, self.eta)
        inner_sum = jnp.sum(one_minus_T_fuzz, axis=0)
        term2 = jnp.sum(gamma * inner_sum)
        return term1 + term2

    @partial(jit, static_argnums=(0,))
    def _iteration_step(self, state: KPFCMState, X: chex.Array) -> tuple[KPFCMState, dict]:
        U_new = self._update_U(X, state.centroids)
        T_new = self._update_T(X, state.centroids, state.gamma)
        centroids_new = self._compute_centroids(X, U_new, T_new, state.centroids)
        gamma_new = self._compute_gamma(X, U_new, centroids_new)
        objective = self._compute_objective(X, U_new, T_new, centroids_new, gamma_new)
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon
        new_state = KPFCMState(centroids_new, U_new, T_new, gamma_new, objective, state.iteration + 1, converged)

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
            U_init = self._update_U(X, centroids_init)
            gamma_init = self._compute_gamma(X, U_init, centroids_init)
            T_init = self._update_T(X, centroids_init, gamma_init)
        elif initial_centroids is not None:
            centroids_init = self._validate_initial_centroids(X, initial_centroids)
            U_init = self._update_U(X, centroids_init)
            gamma_init = self._compute_gamma(X, U_init, centroids_init)
            T_init = self._update_T(X, centroids_init, gamma_init)
        else:
            U_init, T_init, centroids_init = self._initialize(X)
            gamma_init = self._compute_gamma(X, U_init, centroids_init)
        initial_objective = self._compute_objective(X, U_init, T_init, centroids_init, gamma_init)
        initial_state = KPFCMState(centroids_init, U_init, T_init, gamma_init, initial_objective, 0, False)

        final_state, self.history_ = self._run_training(X, initial_state)

        self.centroids_ = final_state.centroids
        self.U_ = final_state.U
        self.T_ = final_state.T
        self.gamma_ = final_state.gamma
        self.n_iter_ = int(final_state.iteration)
        self.objective_ = float(final_state.objective)
        self.objective_history_ = self.history_['objective']
        return self

    def predict(self, X: chex.Array) -> chex.Array:
        self._check_fitted()
        U = self._update_U(X, self.centroids_)
        return jnp.argmax(U, axis=1)

    def predict_proba(self, X: chex.Array) -> chex.Array:
        self._check_fitted()
        X = jnp.asarray(X)
        return self._update_U(X, self.centroids_)

    def get_typicality(self, X: chex.Array) -> chex.Array:
        self._check_fitted()
        X = jnp.asarray(X)
        return self._update_T(X, self.centroids_, self.gamma_)
