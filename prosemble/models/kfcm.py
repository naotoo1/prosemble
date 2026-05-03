"""
JAX-based Kernel Fuzzy C-Means (KFCM) clustering implementation.

This module provides a GPU-accelerated implementation of KFCM using JAX
with JIT compilation for high performance.
"""

from typing import NamedTuple, Self
from functools import partial

import jax
import jax.numpy as jnp
import chex
from jax import jit, lax

from prosemble.core.kernel import batch_gaussian_kernel
from prosemble.models.base import FuzzyClusteringBase, ScanFitMixin


class KFCMState(NamedTuple):
    """Immutable state for KFCM iteration.

    Attributes:
        centroids: Cluster centroids, shape (n_clusters, n_features)
        U: Fuzzy membership matrix, shape (n_samples, n_clusters)
        objective: Current objective function value
        iteration: Current iteration number
        converged: Whether algorithm has converged
    """
    centroids: chex.Array
    U: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class KFCM(ScanFitMixin, FuzzyClusteringBase):
    """
    Kernel Fuzzy C-Means clustering with JAX.

    KFCM uses a Gaussian kernel to map data into a high-dimensional feature space
    where clustering is performed. This allows handling non-linearly separable data.

    Kernel:
        K(x, y) = exp(-||x - y||² / σ²)

    Kernel distance in feature space:
        ||φ(x) - φ(y)||² = 2(1 - K(x, y))

    Algorithm:
    1. Initialize U randomly
    2. Update centroids (kernel-weighted):
       v_j = Σ_i[u_ij^m · K(x_i, v_j) · x_i] / Σ_i[u_ij^m · K(x_i, v_j)]
    3. Update U using kernel distance:
       u_ij = 1 / Σ_k[(1 - K(x_i, v_j)) / (1 - K(x_i, v_k))]^(1/(m-1))
    4. Repeat until convergence

    Objective function:
        J = 2·Σ_i Σ_j [u_ij^m · (1 - K(x_i, v_j))]

    Parameters
    ----------
    n_clusters : int
        Number of clusters (must be >= 2)
    fuzzifier : float, default=2.0
        Fuzziness parameter (must be > 1.0)
    sigma : float, default=1.0
        Kernel bandwidth parameter (must be > 0)
    max_iter : int, default=100
        Maximum number of iterations
    epsilon : float, default=1e-5
        Convergence threshold
    init_method : {'random'}, default='random'
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
    U_ : array, shape (n_samples, n_clusters)
        Final fuzzy membership matrix
    n_iter_ : int
        Number of iterations until convergence
    objective_ : float
        Final objective function value
    objective_history_ : array
        Objective values at each iteration

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from prosemble.models import KFCM
    >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])
    >>> model = KFCM(n_clusters=2, sigma=1.0, random_seed=42)
    >>> model.fit(X)
    >>> labels = model.predict(X)
    """

    _hyperparams = ('fuzzifier', 'sigma', 'init_method')
    _fitted_array_names = ('U_',)

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        sigma: float = 1.0,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        init_method: str = 'random',
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
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if init_method != 'random':
            raise ValueError("init_method must be 'random' for KFCM")

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
        self.sigma = sigma
        self.init_method = init_method

        # Model-specific fitted attributes
        self.U_ = None

    def _initialize(self, X: chex.Array):
        """Initialize U matrix and centroids."""
        n_samples = X.shape[0]

        # Random U matrix (Dirichlet distribution ensures row sums = 1)
        alpha = jnp.ones(self.n_clusters)
        U = jax.random.dirichlet(self.key, alpha, shape=(n_samples,))

        # Random centroids from data
        indices = jax.random.choice(
            self.key, n_samples, shape=(self.n_clusters,), replace=False
        )
        centroids = X[indices]

        return U, centroids

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(
        self, X: chex.Array, U: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute kernel-weighted centroids.

        v_j = Σ_i[u_ij^m · K(x_i, v_j) · x_i] / Σ_i[u_ij^m · K(x_i, v_j)]
        """
        # Compute kernel matrix K(X, centroids)
        K = batch_gaussian_kernel(X, centroids, self.sigma)  # (n_samples, n_clusters)

        # Fuzzify U
        U_fuzz = jnp.power(U, self.fuzzifier)  # (n_samples, n_clusters)

        # Kernel weights
        weights = U_fuzz * K  # (n_samples, n_clusters)

        # Compute centroids
        numerator = weights.T @ X  # (n_clusters, n_features)
        denominator = jnp.sum(weights, axis=0, keepdims=True).T  # (n_clusters, 1)

        denominator = jnp.maximum(denominator, 1e-10)
        centroids_new = numerator / denominator

        return centroids_new

    @partial(jit, static_argnums=(0,))
    def _update_U(self, X: chex.Array, centroids: chex.Array) -> chex.Array:
        """Update fuzzy membership matrix using kernel distance.

        u_ij = 1 / Σ_k[(1 - K(x_i, v_j)) / (1 - K(x_i, v_k))]^(1/(m-1))
        """
        # Compute kernel matrix
        K = batch_gaussian_kernel(X, centroids, self.sigma)  # (n_samples, n_clusters)

        # Kernel distance: 1 - K(x, v)
        kernel_dist = 1.0 - K  # (n_samples, n_clusters)
        kernel_dist = jnp.maximum(kernel_dist, 1e-10)

        # Compute power
        power = 1.0 / (self.fuzzifier - 1.0)

        # Compute distance ratios
        def compute_membership_row(distances_i):
            # distances_i: (n_clusters,)
            ratios = distances_i[:, None] / distances_i[None, :]  # (n_clusters, n_clusters)
            powered_ratios = jnp.power(ratios, power)
            denominators = jnp.sum(powered_ratios, axis=1)  # (n_clusters,)
            memberships = 1.0 / denominators
            return memberships

        U = jax.vmap(compute_membership_row)(kernel_dist)  # (n_samples, n_clusters)

        # Normalize
        U = U / jnp.sum(U, axis=1, keepdims=True)

        return U

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self, X: chex.Array, U: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute KFCM objective function.

        J = 2·Σ_i Σ_j [u_ij^m · (1 - K(x_i, v_j))]
        """
        # Compute kernel matrix
        K = batch_gaussian_kernel(X, centroids, self.sigma)

        # Kernel distance
        kernel_dist = 1.0 - K

        # Fuzzify U
        U_fuzz = jnp.power(U, self.fuzzifier)

        # Weighted sum
        objective = 2.0 * jnp.sum(U_fuzz * kernel_dist)

        return objective

    @partial(jit, static_argnums=(0,))
    def _iteration_step(
        self, state: KFCMState, X: chex.Array
    ) -> tuple[KFCMState, dict]:
        """Single KFCM iteration step."""
        # Update centroids
        centroids_new = self._compute_centroids(X, state.U, state.centroids)

        # Update U
        U_new = self._update_U(X, centroids_new)

        # Compute objective
        objective = self._compute_objective(X, U_new, centroids_new)

        # Check convergence
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon

        new_state = KFCMState(
            centroids=centroids_new,
            U=U_new,
            objective=objective,
            iteration=state.iteration + 1,
            converged=converged
        )

        metrics = {
            'objective': objective,
            'centroid_change': centroid_change,
            'converged': converged
        }

        return new_state, metrics

    def _build_info(self, state, iteration):
        labels = jnp.argmax(state.U, axis=1)
        weights = jnp.max(state.U, axis=1)
        return {
            'centroids': state.centroids, 'labels': labels,
            'weights': weights, 'iteration': iteration,
            'objective': float(state.objective), 'max_iter': self.max_iter,
        }

    def fit(self, X: chex.Array, initial_centroids=None, resume=False) -> Self:
        """Fit KFCM model to data."""
        if resume and initial_centroids is not None:
            raise ValueError("Cannot use both resume=True and initial_centroids")

        X = self._validate_input(X)

        if resume:
            self._check_fitted()
            centroids_init = self.centroids_
            U_init = self._update_U(X, centroids_init)
        elif initial_centroids is not None:
            centroids_init = self._validate_initial_centroids(X, initial_centroids)
            U_init = self._update_U(X, centroids_init)
        else:
            U_init, centroids_init = self._initialize(X)

        initial_objective = self._compute_objective(X, U_init, centroids_init)
        initial_state = KFCMState(
            centroids=centroids_init, U=U_init,
            objective=initial_objective, iteration=0, converged=False
        )

        final_state, self.history_ = self._run_training(X, initial_state)

        # Store results
        self.centroids_ = final_state.centroids
        self.U_ = final_state.U
        self.n_iter_ = int(final_state.iteration)
        self.objective_ = float(final_state.objective)
        self.objective_history_ = self.history_['objective']

        return self

    def predict(self, X: chex.Array) -> chex.Array:
        """Predict cluster labels for new data."""
        self._check_fitted()

        U = self._update_U(X, self.centroids_)
        labels = jnp.argmax(U, axis=1)
        return labels

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """Predict fuzzy membership probabilities."""
        self._check_fitted()
        X = jnp.asarray(X)

        U = self._update_U(X, self.centroids_)
        return U
