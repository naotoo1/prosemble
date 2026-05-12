"""
JAX-based Kernel Possibilistic C-Means (KPCM) clustering implementation.

This module provides a GPU-accelerated implementation of KPCM using JAX
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


class KPCMState(NamedTuple):
    """Immutable state for KPCM iteration.

    Attributes:
        centroids: Cluster centroids, shape (n_clusters, n_features)
        T: Typicality matrix, shape (n_samples, n_clusters)
        gamma: Scale parameters, shape (n_clusters,)
        objective: Current objective function value
        iteration: Current iteration number
        converged: Whether algorithm has converged
    """
    centroids: chex.Array
    T: chex.Array
    gamma: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class KPCM(ScanFitMixin, FuzzyClusteringBase):
    """
    Kernel Possibilistic C-Means clustering with JAX.

    KPCM extends PCM to kernel space using Gaussian kernel, allowing handling
    of non-linearly separable data while maintaining possibilistic properties.

    Kernel::

        K(x, y) = exp(-||x - y||² / σ²)

    Kernel distance::

        d_K(x, v) = 2(1 - K(x, v))

    Algorithm:

    1. Initialize using KFCM
    2. Compute gamma parameters
    3. Update typicality matrix T
    4. Update centroids (kernel-weighted)
    5. Repeat until convergence

    Objective function::

        J = Σ_i Σ_j [t_ij^m · d_K(x_i, v_j)] + Σ_j[γ_j · Σ_i(1 - t_ij)^m]

    Parameters
    ----------
    n_clusters : int
        Number of clusters (must be >= 2)
    fuzzifier : float, default=2.0
        Fuzziness parameter (must be > 1.0)
    k : float, default=1.0
        Scaling parameter for gamma (must be > 0)
    sigma : float, default=1.0
        Kernel bandwidth parameter (must be > 0)
    max_iter : int, default=100
        Maximum number of iterations
    epsilon : float, default=1e-5
        Convergence threshold
    init_method : {'kfcm'}, default='kfcm'
        Initialization method
    random_seed : int, default=42
        Random seed for reproducibility
    plot_steps : bool, default=False
        Whether to visualize clustering progress
    show_confidence : bool, default=True
        Whether to show confidence in visualization
    show_pca_variance : bool, default=True
        Whether to show PCA variance in visualization
    save_plot_path : str, optional
        Path to save final plot

    Attributes
    ----------
    centroids_ : array, shape (n_clusters, n_features)
        Final cluster centroids
    T_ : array, shape (n_samples, n_clusters)
        Final typicality matrix
    gamma_ : array, shape (n_clusters,)
        Final scale parameters
    n_iter_ : int
        Number of iterations until convergence
    objective_ : float
        Final objective function value
    objective_history_ : array
        Objective values at each iteration

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from prosemble.models import KPCM
    >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])
    >>> model = KPCM(n_clusters=2, sigma=1.0, random_seed=42)
    >>> model.fit(X)
    >>> labels = model.predict(X)

    See Also
    --------
    FuzzyClusteringBase : Full list of base parameters (distance_fn,
        patience, restore_best, callbacks, etc.).
    """

    _hyperparams = ('fuzzifier', 'sigma', 'init_method')
    _fitted_array_names = ('T_', 'gamma_')

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        k: float = 1.0,
        sigma: float = 1.0,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        init_method: str = 'kfcm',
        random_seed: int = 42,
        plot_steps: bool = False,
        show_confidence: bool = True,
        show_pca_variance: bool = True,
        save_plot_path: str | None = None,
        **kwargs
    ):
        # Validate model-specific parameters
        if fuzzifier <= 1.0:
            raise ValueError("fuzzifier must be > 1.0")
        if k <= 0:
            raise ValueError("k must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if init_method != 'kfcm':
            raise ValueError("init_method must be 'kfcm' for KPCM")

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
        self.k = k
        self.sigma = sigma
        self.init_method = init_method

        # Model-specific fitted attributes
        self.T_ = None
        self.gamma_ = None

    def _initialize(self, X: chex.Array):
        """Initialize using KFCM."""
        # Use KFCM to initialize
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

        # Initialize gamma from U
        gamma = self._compute_gamma(X, U, centroids)

        return U, centroids, gamma

    @partial(jit, static_argnums=(0,))
    def _compute_gamma(
        self, X: chex.Array, U: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute scale parameters.

        γ_j = k·Σ_i(u_ij^m · d_K(x_i, v_j)) / Σ_i(u_ij^m)
        """
        # Compute kernel matrix
        K = batch_gaussian_kernel(X, centroids, self.sigma)

        # Kernel distance
        kernel_dist = 2.0 * (1.0 - K)

        # Fuzzify U
        U_fuzz = jnp.power(U, self.fuzzifier)

        # Compute gamma
        numerator = jnp.sum(U_fuzz * kernel_dist, axis=0)
        denominator = jnp.sum(U_fuzz, axis=0)

        gamma = self.k * numerator / denominator

        return gamma

    @partial(jit, static_argnums=(0,))
    def _update_T(
        self, X: chex.Array, centroids: chex.Array, gamma: chex.Array
    ) -> chex.Array:
        """Update typicality matrix.

        t_ij = 1 / (1 + (d_K(x_i, v_j)/γ_j)^(1/(m-1)))
        """
        # Compute kernel distance
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)
        kernel_dist = jnp.maximum(kernel_dist, 1e-10)

        # Compute power
        power = 1.0 / (self.fuzzifier - 1.0)

        # Compute typicality
        ratio = kernel_dist / gamma[None, :]
        T = 1.0 / (1.0 + jnp.power(ratio, power))

        return T

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(
        self, X: chex.Array, T: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute kernel-weighted centroids.

        v_j = Σ_i[t_ij^m · K(x_i, v_j) · x_i] / Σ_i[t_ij^m · K(x_i, v_j)]
        """
        # Compute kernel matrix
        K = batch_gaussian_kernel(X, centroids, self.sigma)

        # Fuzzify T
        T_fuzz = jnp.power(T, self.fuzzifier)

        # Kernel weights
        weights = T_fuzz * K

        # Compute centroids
        numerator = weights.T @ X
        denominator = jnp.sum(weights, axis=0, keepdims=True).T

        denominator = jnp.maximum(denominator, 1e-10)
        centroids_new = numerator / denominator

        return centroids_new

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self, X: chex.Array, T: chex.Array, centroids: chex.Array, gamma: chex.Array
    ) -> chex.Array:
        """Compute KPCM objective function.

        J = Σ_i Σ_j [t_ij^m · d_K(x_i, v_j)] + Σ_j[γ_j · Σ_i(1 - t_ij)^m]
        """
        # Compute kernel distance
        K = batch_gaussian_kernel(X, centroids, self.sigma)
        kernel_dist = 2.0 * (1.0 - K)

        # Fuzzify T
        T_fuzz = jnp.power(T, self.fuzzifier)

        # First term
        term1 = jnp.sum(T_fuzz * kernel_dist)

        # Second term
        one_minus_T = 1.0 - T
        one_minus_T_fuzz = jnp.power(one_minus_T, self.fuzzifier)
        inner_sum = jnp.sum(one_minus_T_fuzz, axis=0)
        term2 = jnp.sum(gamma * inner_sum)

        objective = term1 + term2

        return objective

    @partial(jit, static_argnums=(0,))
    def _iteration_step(self, state: KPCMState, X: chex.Array) -> tuple[KPCMState, dict]:
        """Single KPCM iteration step."""
        # Update T
        T_new = self._update_T(X, state.centroids, state.gamma)

        # Update centroids
        centroids_new = self._compute_centroids(X, T_new, state.centroids)

        # Compute objective
        objective = self._compute_objective(X, T_new, centroids_new, state.gamma)

        # Check convergence
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon

        new_state = KPCMState(
            centroids=centroids_new,
            T=T_new,
            gamma=state.gamma,
            objective=objective,
            iteration=state.iteration + 1,
            converged=converged
        )

        metrics = {
            'objective': new_state.objective,
            'centroid_change': centroid_change,
            'converged': new_state.converged,
        }

        return new_state, metrics

    def _build_info(self, state, iteration):
        labels = jnp.argmax(state.T, axis=1)
        weights = jnp.max(state.T, axis=1)
        return {
            'centroids': state.centroids, 'labels': labels,
            'weights': weights, 'iteration': iteration,
            'objective': float(state.objective), 'max_iter': self.max_iter,
        }

    def fit(self, X: chex.Array, initial_centroids=None, resume=False) -> Self:
        """Fit KPCM model to data."""
        if resume and initial_centroids is not None:
            raise ValueError("Cannot use both resume=True and initial_centroids")

        X = self._validate_input(X)

        if resume:
            self._check_fitted()
            centroids_init = self.centroids_
            T_init = self.T_
            gamma_init = self.gamma_
        elif initial_centroids is not None:
            centroids_init = self._validate_initial_centroids(X, initial_centroids)
            # Derive gamma from uniform U, then T
            U_uniform = jnp.ones((X.shape[0], self.n_clusters)) / self.n_clusters
            gamma_init = self._compute_gamma(X, U_uniform, centroids_init)
            T_init = self._update_T(X, centroids_init, gamma_init)
        else:
            # Initialize
            U_init, centroids_init, gamma_init = self._initialize(X)
            T_init = self._update_T(X, centroids_init, gamma_init)
        initial_objective = self._compute_objective(X, T_init, centroids_init, gamma_init)
        initial_state = KPCMState(
            centroids=centroids_init, T=T_init, gamma=gamma_init,
            objective=initial_objective, iteration=0, converged=False
        )

        final_state, self.history_ = self._run_training(X, initial_state)

        # Store results
        self.centroids_ = final_state.centroids
        self.T_ = final_state.T
        self.gamma_ = final_state.gamma
        self.n_iter_ = int(final_state.iteration)
        self.objective_ = float(final_state.objective)
        self.objective_history_ = self.history_['objective']

        return self

    def predict(self, X: chex.Array) -> chex.Array:
        """Predict cluster labels for new data."""
        self._check_fitted()

        T = self._update_T(X, self.centroids_, self.gamma_)
        labels = jnp.argmax(T, axis=1)
        return labels

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """Predict typicality values."""
        self._check_fitted()
        X = jnp.asarray(X)

        T = self._update_T(X, self.centroids_, self.gamma_)
        return T
