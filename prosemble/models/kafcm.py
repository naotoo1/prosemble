"""
JAX-based Kernel Adaptive Fuzzy C-Means (KAFCM) clustering implementation.

This module provides a GPU-accelerated implementation of KAFCM using JAX
with JIT compilation for high performance.
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


class KAFCMState(NamedTuple):
    """Immutable state for KAFCM iteration."""
    centroids: chex.Array
    U: chex.Array
    T: chex.Array
    gamma: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class KAFCM(ScanFitMixin, FuzzyClusteringBase):
    """
    Kernel Adaptive Fuzzy C-Means clustering with JAX.

    KAFCM extends AFCM to kernel space, combining fuzzy and possibilistic
    approaches with kernel-based distance measures.

    Kernel distance: d_K(x, v) = 2(1 - K(x, v))

    Algorithm:

    1. Initialize U using KFCM
    2. Compute gamma parameters using kernel distance
    3. Update T using exponential kernel update
    4. Update U using standard KFCM rule
    5. Update centroids (kernel-weighted with combined weights)
    6. Repeat until convergence

    Parameters
    ----------
    fuzzifier : float, default=2.0
        Fuzziness parameter (must be > 1.0).
    a : float, default=1.0
        Weight for fuzzy membership (must be > 0).
    b : float, default=1.0
        Weight for typicality (must be > 0).
    k : float, default=1.0
        Scaling parameter for gamma (must be > 0).
    sigma : float, default=1.0
        Kernel bandwidth parameter (must be > 0).
    init_method : {'kfcm'}, default='kfcm'
        Initialization method.
    n_clusters : int
        Number of clusters (must be >= 2).
    max_iter : int
        Maximum number of iterations.
    epsilon : float
        Convergence threshold.
    random_seed : int
        Random seed for reproducibility.
    distance_fn : callable, optional
        Pairwise distance function. Default: squared Euclidean.
    patience : int, optional
        Epochs with no improvement before early stopping. Default: None.
    restore_best : bool
        If True, restore centroids from the lowest-objective epoch.
        Default: False.
    plot_steps : bool
        Whether to visualize clustering progress. Default: False.
    show_confidence : bool
        Whether to show confidence in visualization. Default: True.
    show_pca_variance : bool
        Whether to show PCA variance in visualization. Default: True.
    save_plot_path : str, optional
        Path to save final plot.
    callbacks : list, optional
        List of Callback objects for monitoring/visualization.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from prosemble.models import KAFCM
    >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])
    >>> model = KAFCM(n_clusters=2, sigma=1.0, random_seed=42)
    >>> model.fit(X)
    >>> labels = model.predict(X)
    """

    _hyperparams = ('fuzzifier', 'sigma', 'a', 'b', 'k', 'init_method')
    _fitted_array_names = ('U_', 'T_', 'gamma_')

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        a: float = 1.0,
        b: float = 1.0,
        k: float = 1.0,
        sigma: float = 1.0,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        init_method: str = 'kfcm',
        random_seed: int = 42,
        distance_fn=None,
        patience: int | None = None,
        restore_best: bool = False,
        plot_steps: bool = False,
        show_confidence: bool = True,
        show_pca_variance: bool = True,
        save_plot_path: str | None = None,
        callbacks=None,
    ):
        # Validate model-specific parameters
        if fuzzifier <= 1.0:
            raise ValueError("fuzzifier must be > 1.0")
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
            distance_fn=distance_fn,
            patience=patience,
            restore_best=restore_best,
            plot_steps=plot_steps,
            show_confidence=show_confidence,
            show_pca_variance=show_pca_variance,
            save_plot_path=save_plot_path,
            callbacks=callbacks,
        )

        self.fuzzifier = fuzzifier
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
        """Initialize using KFCM."""
        n_samples = X.shape[0]
        kfcm = KFCM(
            n_clusters=self.n_clusters,
            fuzzifier=self.fuzzifier,
            sigma=self.sigma,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            random_seed=self.random_seed,
            plot_steps=False
        )
        kfcm.fit(X)
        U = kfcm.U_
        centroids = kfcm.centroids_
        T = jnp.zeros((n_samples, self.n_clusters))
        return U, T, centroids

    @partial(jit, static_argnums=(0,))
    def _compute_gamma(
        self, X: chex.Array, U: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute gamma using kernel distance."""
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        U_fuzz = jnp.power(U, self.fuzzifier)
        numerator = jnp.sum(U_fuzz * kernel_dist, axis=0)
        denominator = jnp.sum(U_fuzz, axis=0)
        gamma = self.k * numerator / denominator
        return gamma

    @partial(jit, static_argnums=(0,))
    def _update_T(
        self, X: chex.Array, centroids: chex.Array, gamma: chex.Array
    ) -> chex.Array:
        """Update T: t_ij = exp(-b·d_K(x_i, v_j)/γ_j)"""
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        kernel_dist = jnp.maximum(kernel_dist, 1e-10)
        ratio = self.b * kernel_dist / gamma[None, :]
        T = jnp.exp(-ratio)
        return T

    @partial(jit, static_argnums=(0,))
    def _update_U(self, X: chex.Array, centroids: chex.Array) -> chex.Array:
        """Update U using kernel distance."""
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 1.0 - K
        kernel_dist = jnp.maximum(kernel_dist, 1e-10)
        power = 1.0 / (self.fuzzifier - 1.0)

        def compute_membership_row(distances_i):
            ratios = distances_i[:, None] / distances_i[None, :]
            powered_ratios = jnp.power(ratios, power)
            denominators = jnp.sum(powered_ratios, axis=1)
            memberships = 1.0 / denominators
            return memberships

        U = jax.vmap(compute_membership_row)(kernel_dist)
        U = U / jnp.sum(U, axis=1, keepdims=True)
        return U

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(
        self, X: chex.Array, U: chex.Array, T: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute centroids: kernel-weighted with a·U^m + b·T"""
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        U_fuzz = jnp.power(U, self.fuzzifier)
        weights = (self.a * U_fuzz + self.b * T) * K
        numerator = weights.T @ X
        denominator = jnp.sum(weights, axis=0, keepdims=True).T
        denominator = jnp.maximum(denominator, 1e-10)
        centroids_new = numerator / denominator
        return centroids_new

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self, X: chex.Array, U: chex.Array, T: chex.Array,
        centroids: chex.Array, gamma: chex.Array
    ) -> chex.Array:
        """Compute KAFCM objective."""
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        U_fuzz = jnp.power(U, self.fuzzifier)
        weights = self.a * U_fuzz + self.b * T
        term1 = jnp.sum(kernel_dist * weights)
        T_safe = jnp.maximum(T, 1e-10)
        entropy_like = T * jnp.log(T_safe) - T
        inner_sum = jnp.sum(entropy_like, axis=0)
        term2 = jnp.sum(gamma * inner_sum)
        objective = term1 + term2
        return objective

    @partial(jit, static_argnums=(0,))
    def _iteration_step(self, state: KAFCMState, X: chex.Array) -> tuple[KAFCMState, dict]:
        """Single iteration."""
        T_new = self._update_T(X, state.centroids, state.gamma)
        U_new = self._update_U(X, state.centroids)
        centroids_new = self._compute_centroids(X, U_new, T_new, state.centroids)
        gamma_new = self._compute_gamma(X, U_new, centroids_new)
        objective = self._compute_objective(X, U_new, T_new, centroids_new, gamma_new)
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon
        new_state = KAFCMState(centroids_new, U_new, T_new, gamma_new, objective, state.iteration + 1, converged)

        metrics = {
            'objective': new_state.objective,
            'centroid_change': centroid_change,
            'converged': new_state.converged,
        }

        return new_state, metrics

    def _build_info(self, state, iteration):
        labels = jnp.argmax(state.U, axis=1)
        weights = jnp.max(state.U * state.T, axis=1)
        return {
            'centroids': state.centroids, 'labels': labels,
            'weights': weights, 'iteration': iteration,
            'objective': float(state.objective), 'max_iter': self.max_iter,
        }

    def fit(self, X: chex.Array, initial_centroids=None, resume=False) -> Self:
        """Fit model."""
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
        initial_state = KAFCMState(centroids_init, U_init, T_init, gamma_init, initial_objective, 0, False)

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
        """Predict labels."""
        self._check_fitted()
        U = self._update_U(X, self.centroids_)
        return jnp.argmax(U, axis=1)

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """Predict probabilities."""
        self._check_fitted()
        X = jnp.asarray(X)
        return self._update_U(X, self.centroids_)

    def get_typicality(self, X: chex.Array) -> chex.Array:
        """Get typicality."""
        self._check_fitted()
        X = jnp.asarray(X)
        return self._update_T(X, self.centroids_, self.gamma_)
