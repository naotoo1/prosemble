"""
JAX-based Kernel Improved Possibilistic C-Means 2 (KIPCM2) clustering implementation.
"""

from typing import NamedTuple, Self
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import chex
from jax import jit
from prosemble.core.kernel import batch_gaussian_kernel
from prosemble.models.kfcm import KFCM
from prosemble.models.base import FuzzyClusteringBase


class KIPCM2State(NamedTuple):
    centroids: chex.Array
    U: chex.Array
    T: chex.Array
    gamma: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool
    phase: int


class KIPCM2(FuzzyClusteringBase):
    """Kernel Improved Possibilistic C-Means 2 with JAX.

    KIPCM2 uses exponential T update and modified objective in kernel space.

    See Also
    --------
    FuzzyClusteringBase : Full list of base parameters (distance_fn,
        patience, restore_best, callbacks, etc.).
    """

    _hyperparams = ('fuzzifier', 'tipifier', 'sigma', 'init_method')
    _fitted_array_names = ('U_', 'T_', 'gamma_')

    def __init__(self, n_clusters: int, fuzzifier: float = 2.0, tipifier: float = 2.0,
                 sigma: float = 1.0, max_iter: int = 100, epsilon: float = 1e-5,
                 init_method: str = 'kfcm', random_seed: int = 42, plot_steps: bool = False,
                 show_confidence: bool = True, show_pca_variance: bool = True,
                 save_plot_path: str | None = None):
        # Validate model-specific parameters
        if fuzzifier <= 1.0:
            raise ValueError("fuzzifier must be > 1.0")
        if tipifier <= 1.0:
            raise ValueError("tipifier must be > 1.0")
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
            save_plot_path=save_plot_path,
        )

        self.fuzzifier = fuzzifier
        self.tipifier = tipifier
        self.sigma = sigma
        self.init_method = init_method

        # Fitted attributes
        self.U_ = None
        self.T_ = None
        self.gamma_ = None


    def _initialize_phase0(self, X: chex.Array):
        n_samples = X.shape[0]
        kfcm = KFCM(self.n_clusters, self.fuzzifier, self.sigma, self.max_iter,
                       self.epsilon, 'random', self.random_seed, False)
        kfcm.fit(X)
        U = kfcm.U_
        centroids = kfcm.centroids_
        T = jnp.zeros((n_samples, self.n_clusters))
        return U, T, centroids

    @partial(jit, static_argnums=(0,))
    def _compute_gamma_phase0(self, X: chex.Array, U: chex.Array, centroids: chex.Array) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        U_fuzz = jnp.power(U, self.fuzzifier)
        numerator = jnp.sum(U_fuzz * kernel_dist, axis=0)
        denominator = jnp.maximum(jnp.sum(U_fuzz, axis=0), 1e-10)
        return numerator / denominator

    @partial(jit, static_argnums=(0,))
    def _compute_gamma_phase1(self, X: chex.Array, U: chex.Array, T: chex.Array, centroids: chex.Array) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        U_fuzz = jnp.power(U, self.fuzzifier)
        T_fuzz = jnp.power(T, self.tipifier)
        prod = U_fuzz * T_fuzz
        numerator = jnp.sum(prod * kernel_dist, axis=0)
        denominator = jnp.maximum(jnp.sum(prod, axis=0), 1e-10)
        return numerator / denominator

    @partial(jit, static_argnums=(0,))
    def _update_T(self, X: chex.Array, centroids: chex.Array, gamma: chex.Array) -> chex.Array:
        """Exponential T update: t_ij = exp(-d_K(x_i, v_j)/γ_j)"""
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        kernel_dist = jnp.maximum(kernel_dist, 1e-10)
        ratio = kernel_dist / gamma[None, :]
        return jnp.exp(-ratio)

    @partial(jit, static_argnums=(0,))
    def _update_U(self, X: chex.Array, centroids: chex.Array, gamma: chex.Array) -> chex.Array:
        """Modified U update for KIPCM2."""
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        kernel_dist = jnp.maximum(kernel_dist, 1e-10)
        ratio = kernel_dist / gamma[None, :]
        exp_term = jnp.exp(-ratio)
        modified_dist = gamma[None, :] * (1.0 - exp_term)
        modified_dist = jnp.maximum(modified_dist, 1e-10)
        base_values = 1.0 / modified_dist
        power = 2.0 / (self.fuzzifier - 1.0)
        powered_values = jnp.power(base_values, power)
        denominators = jnp.sum(powered_values, axis=1, keepdims=True)
        return powered_values / denominators

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(self, X: chex.Array, U: chex.Array, T: chex.Array, centroids: chex.Array) -> chex.Array:
        """Centroids: kernel-weighted with U^m · T (T not raised to power!)"""
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        U_fuzz = jnp.power(U, self.fuzzifier)
        weights = (U_fuzz * T) * K
        numerator = weights.T @ X
        denominator = jnp.sum(weights, axis=0, keepdims=True).T
        return numerator / jnp.maximum(denominator, 1e-10)

    @partial(jit, static_argnums=(0,))
    def _compute_objective(self, X: chex.Array, U: chex.Array, T: chex.Array,
                          centroids: chex.Array, gamma: chex.Array) -> chex.Array:
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        U_fuzz = jnp.power(U, self.fuzzifier)
        term1 = jnp.sum(U_fuzz * T * kernel_dist)
        T_safe = jnp.maximum(T, 1e-10)
        entropy_like = T * jnp.log(T_safe) - T + 1.0
        inner_sum = jnp.sum(entropy_like * U_fuzz, axis=0)
        term2 = jnp.sum(gamma * inner_sum)
        return term1 + term2

    @partial(jit, static_argnums=(0,))
    def _iteration_step(self, state: KIPCM2State, X: chex.Array) -> KIPCM2State:
        T_new = self._update_T(X, state.centroids, state.gamma)
        U_new = self._update_U(X, state.centroids, state.gamma)
        centroids_new = self._compute_centroids(X, U_new, T_new, state.centroids)
        objective = self._compute_objective(X, U_new, T_new, centroids_new, state.gamma)
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon
        return KIPCM2State(centroids_new, U_new, T_new, state.gamma, objective, state.iteration + 1, converged, state.phase)

    def _build_info(self, state, iteration):
        labels = jnp.argmax(state.U, axis=1)
        weights = jnp.max(state.U * state.T, axis=1)
        return {
            'centroids': state.centroids, 'labels': labels,
            'weights': weights, 'iteration': iteration,
            'objective': float(state.objective), 'max_iter': self.max_iter,
        }

    def _run_phase(self, X: chex.Array, U_init: chex.Array, T_init: chex.Array,
                   centroids_init: chex.Array, gamma_init: chex.Array, phase: int):
        initial_objective = self._compute_objective(X, U_init, T_init, centroids_init, gamma_init)
        state = KIPCM2State(centroids_init, U_init, T_init, gamma_init, initial_objective, 0, False, phase)
        states_history = [state]
        for i in range(self.max_iter):
            self._notify_iteration(self._build_info(state, state.iteration))
            state = self._iteration_step(state, X)
            states_history.append(state)
            if state.converged:
                break
        return state, states_history

    def fit(self, X: chex.Array, initial_centroids=None, resume=False) -> Self:
        if resume and initial_centroids is not None:
            raise ValueError("Cannot use both resume=True and initial_centroids")

        X = self._validate_input(X)
        self._notify_fit_start(X)

        if resume:
            self._check_fitted()
            gamma_1 = self._compute_gamma_phase1(X, self.U_, self.T_, self.centroids_)
            state_final, history_phase1 = self._run_phase(X, self.U_, self.T_, self.centroids_, gamma_1, phase=1)
            self._notify_fit_end(self._build_info(state_final, state_final.iteration))
            all_objectives = [s.objective for s in history_phase1]
            self.objective_history_ = jnp.array(all_objectives)
        else:
            if initial_centroids is not None:
                centroids_init = self._validate_initial_centroids(X, initial_centroids)
                U_uniform = jnp.ones((X.shape[0], self.n_clusters)) / self.n_clusters
                gamma_0 = self._compute_gamma_phase0(X, U_uniform, centroids_init)
                T_init = self._update_T(X, centroids_init, gamma_0)
                U_init = self._update_U(X, centroids_init, gamma_0)
            else:
                U_init, T_init, centroids_init = self._initialize_phase0(X)

            gamma_0 = self._compute_gamma_phase0(X, U_init, centroids_init)
            state_phase0, history_phase0 = self._run_phase(X, U_init, T_init, centroids_init, gamma_0, phase=0)
            gamma_1 = self._compute_gamma_phase1(X, state_phase0.U, state_phase0.T, state_phase0.centroids)
            state_final, history_phase1 = self._run_phase(X, state_phase0.U, state_phase0.T, state_phase0.centroids, gamma_1, phase=1)
            self._notify_fit_end(self._build_info(state_final, state_final.iteration))
            all_objectives = [s.objective for s in history_phase0] + [s.objective for s in history_phase1]
            self.objective_history_ = jnp.array(all_objectives)
        self.centroids_ = state_final.centroids
        self.U_ = state_final.U
        self.T_ = state_final.T
        self.gamma_ = state_final.gamma
        self.n_iter_ = int(state_final.iteration)
        self.objective_ = float(state_final.objective)
        return self

    def predict(self, X: chex.Array) -> chex.Array:
        self._check_fitted()
        T = self._update_T(X, self.centroids_, self.gamma_)
        U = self._update_U(X, self.centroids_, self.gamma_)
        return jnp.argmax(U, axis=1)

    def predict_proba(self, X: chex.Array) -> chex.Array:
        self._check_fitted()
        X = jnp.asarray(X)
        return self._update_U(X, self.centroids_, self.gamma_)

    def get_typicality(self, X: chex.Array) -> chex.Array:
        self._check_fitted()
        X = jnp.asarray(X)
        return self._update_T(X, self.centroids_, self.gamma_)
