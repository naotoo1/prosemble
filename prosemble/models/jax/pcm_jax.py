"""
JAX-based Possibilistic C-Means (PCM) clustering.

This module provides a GPU-accelerated implementation of Possibilistic C-Means
using JAX for automatic differentiation and JIT compilation.

PCM extends FCM by introducing typicality values that represent the degree to which
a data point belongs to a cluster, independent of other clusters. This makes PCM
less sensitive to outliers and noise compared to FCM.

Mathematical formulation:
    Objective function:
        J = Σᵢ Σⱼ tᵢⱼᵐ ||xᵢ - vⱼ||² + Σⱼ γⱼ Σᵢ (1 - tᵢⱼ)ᵐ

    where:
        - tᵢⱼ is the typicality of point xᵢ to cluster j
        - vⱼ is the centroid of cluster j
        - m is the fuzzifier (m > 1)
        - γⱼ is a scale parameter for cluster j

    Update equations:
        Centroids:
            vⱼ = (Σᵢ tᵢⱼᵐ xᵢ) / (Σᵢ tᵢⱼᵐ)

        Gamma:
            γⱼ = k · (Σᵢ tᵢⱼᵐ ||xᵢ - vⱼ||²) / (Σᵢ tᵢⱼᵐ)

        Typicality:
            tᵢⱼ = 1 / (1 + (||xᵢ - vⱼ||² / γⱼ)^(1/(m-1)))

References:
    Krishnapuram, R., & Keller, J. M. (1993).
    A possibilistic approach to clustering.
    IEEE Transactions on Fuzzy Systems, 1(2), 98-110.
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT

from typing import Optional, Tuple, NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
import chex

from prosemble.core.distance_jax import (
    euclidean_distance_matrix,
    squared_euclidean_distance_matrix
)
from .fcm_jax import FCM_JAX


class PCMState(NamedTuple):
    """Immutable state for PCM training using JAX.

    Attributes:
        centroids: Cluster centroids, shape (c, d)
        T: Typicality matrix, shape (n, c)
        gamma: Scale parameters for each cluster, shape (c,)
        objective: Current objective function value
        iteration: Current iteration number
        converged: Whether the algorithm has converged
    """
    centroids: chex.Array
    T: chex.Array
    gamma: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class PCM_JAX:
    """
    JAX-based Possibilistic C-Means clustering with GPU acceleration.

    PCM is a clustering algorithm that assigns typicality values to data points,
    representing the degree to which they belong to each cluster. Unlike FCM,
    the typicality of a point to one cluster is independent of its typicality
    to other clusters.

    Parameters:
        n_clusters: int
            Number of clusters to form.

        fuzzifier: float, default=2.0
            Fuzzification parameter (m > 1). Higher values result in fuzzier clusters.

        k: float, default=1.0
            Parameter for gamma computation. Typical values are in [0.01, 1.0].
            Lower values make the algorithm more sensitive to outliers.

        max_iter: int, default=100
            Maximum number of iterations.

        epsilon: float, default=1e-5
            Convergence tolerance. Algorithm stops when centroid change is below this.

        init_method: {'fcm', 'random'}, default='fcm'
            Initialization method:
            - 'fcm': Initialize using FCM results (recommended)
            - 'random': Random initialization

        random_seed: int, optional
            Random seed for reproducibility.

    Attributes:
        centroids_: ndarray of shape (n_clusters, n_features)
            Cluster centroids after fitting.

        T_: ndarray of shape (n_samples, n_clusters)
            Typicality matrix after fitting.

        gamma_: ndarray of shape (n_clusters,)
            Scale parameters for each cluster.

        n_iter_: int
            Number of iterations run.

        objective_: float
            Final objective function value.

    Examples:
        >>> import jax.numpy as jnp
        >>> from prosemble.models.jax import PCM_JAX
        >>>
        >>> # Generate sample data
        >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
        >>>
        >>> # Fit PCM model
        >>> model = PCM_JAX(n_clusters=2, fuzzifier=2.0, k=1.0)
        >>> model.fit(X)
        >>>
        >>> # Get cluster assignments
        >>> labels = model.predict(X)
        >>>
        >>> # Get typicality values
        >>> typicalities = model.predict_proba(X)

    Notes:
        - PCM is less sensitive to outliers than FCM because typicality values
          are computed independently for each cluster.
        - The parameter k controls the sensitivity to outliers. Smaller values
          make the algorithm more sensitive.
        - Initialization from FCM (init_method='fcm') is recommended as it provides
          better starting points than random initialization.
        - All computations are JIT-compiled and can run on GPU if available.
    """

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        k: float = 1.0,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        init_method: str = 'fcm',
        random_seed: Optional[int] = None
    ):
        # Validate parameters
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        if fuzzifier <= 1.0:
            raise ValueError(f"fuzzifier must be > 1.0, got {fuzzifier}")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if init_method not in ['fcm', 'random']:
            raise ValueError(f"init_method must be 'fcm' or 'random', got {init_method}")

        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.k = k
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.init_method = init_method

        # Initialize random key
        if random_seed is None:
            random_seed = 42
        self.key = jax.random.PRNGKey(random_seed)

        # Attributes set during fitting
        self.centroids_ = None
        self.T_ = None
        self.gamma_ = None
        self.n_iter_ = None
        self.objective_ = None
        self._objective_history = None

    def _initialize_from_fcm(self, X: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Initialize centroids and typicality matrix using FCM.

        Args:
            X: Data matrix of shape (n, d)

        Returns:
            centroids: Initial centroids of shape (c, d)
            T: Initial typicality matrix of shape (n, c)
        """
        # Run FCM to get initial centroids and membership matrix
        fcm = FCM_JAX(
            n_clusters=self.n_clusters,
            fuzzifier=self.fuzzifier,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            random_seed=int(self.key[0])
        )
        fcm.fit(X)

        # Use FCM centroids and membership as initial typicality
        centroids = fcm.centroids_
        T = fcm.U_  # Use membership as initial typicality

        return centroids, T

    def _initialize_random(self, X: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """
        Initialize centroids and typicality matrix randomly.

        Args:
            X: Data matrix of shape (n, d)

        Returns:
            centroids: Initial centroids of shape (c, d)
            T: Initial typicality matrix of shape (n, c)
        """
        n_samples, n_features = X.shape

        # Random centroids from data points
        key1, key2, self.key = jax.random.split(self.key, 3)
        indices = jax.random.choice(key1, n_samples, shape=(self.n_clusters,), replace=False)
        centroids = X[indices]

        # Random typicality matrix (using Dirichlet distribution for valid probabilities)
        alpha = jnp.ones(self.n_clusters)
        T = jax.random.dirichlet(key2, alpha, shape=(n_samples,))

        return centroids, T

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(self, X: chex.Array, T: chex.Array) -> chex.Array:
        """
        Compute cluster centroids from typicality matrix.

        Vectorized computation:
            v_j = (Σᵢ tᵢⱼᵐ xᵢ) / (Σᵢ tᵢⱼᵐ)

        Using matrix operations:
            V = (T^m)^T @ X / sum(T^m, axis=0)

        Args:
            X: Data matrix of shape (n, d)
            T: Typicality matrix of shape (n, c)

        Returns:
            centroids: Cluster centroids of shape (c, d)
        """
        # Fuzzify typicality matrix: T^m
        T_fuzz = jnp.power(T, self.fuzzifier)  # (n, c)

        # Numerator: (T^m)^T @ X = (c, n) @ (n, d) = (c, d)
        numerator = T_fuzz.T @ X

        # Denominator: sum of each column of T^m = (c,)
        denominator = jnp.sum(T_fuzz, axis=0, keepdims=True).T  # (c, 1)

        # Compute centroids with numerical stability
        centroids = numerator / jnp.maximum(denominator, 1e-10)

        return centroids

    @partial(jit, static_argnums=(0,))
    def _compute_gamma(self, X: chex.Array, T: chex.Array, centroids: chex.Array) -> chex.Array:
        """
        Compute gamma parameters for each cluster.

        Vectorized computation:
            γⱼ = k · (Σᵢ tᵢⱼᵐ ||xᵢ - vⱼ||²) / (Σᵢ tᵢⱼᵐ)

        Args:
            X: Data matrix of shape (n, d)
            T: Typicality matrix of shape (n, c)
            centroids: Cluster centroids of shape (c, d)

        Returns:
            gamma: Scale parameters of shape (c,)
        """
        # Compute squared distances: (n, c)
        D_sq = squared_euclidean_distance_matrix(X, centroids)

        # Fuzzify typicality: (n, c)
        T_fuzz = jnp.power(T, self.fuzzifier)

        # Weighted distances: element-wise multiply and sum over samples
        # numerator = Σᵢ tᵢⱼᵐ ||xᵢ - vⱼ||²
        numerator = jnp.sum(T_fuzz * D_sq, axis=0)  # (c,)

        # denominator = Σᵢ tᵢⱼᵐ
        denominator = jnp.sum(T_fuzz, axis=0)  # (c,)

        # Compute gamma with numerical stability
        gamma = self.k * numerator / jnp.maximum(denominator, 1e-10)

        return gamma

    @partial(jit, static_argnums=(0,))
    def _update_typicality(
        self,
        X: chex.Array,
        centroids: chex.Array,
        gamma: chex.Array
    ) -> chex.Array:
        """
        Update typicality matrix.

        Vectorized computation:
            tᵢⱼ = 1 / (1 + (||xᵢ - vⱼ||² / γⱼ)^(1/(m-1)))

        Args:
            X: Data matrix of shape (n, d)
            centroids: Cluster centroids of shape (c, d)
            gamma: Scale parameters of shape (c,)

        Returns:
            T: Updated typicality matrix of shape (n, c)
        """
        # Compute squared distances: (n, c)
        D_sq = squared_euclidean_distance_matrix(X, centroids)

        # Compute exponent
        exponent = 1.0 / (self.fuzzifier - 1.0)

        # Compute denominator: (D²ᵢⱼ / γⱼ)^(1/(m-1))
        # Add small epsilon to gamma to avoid division by zero
        ratio = D_sq / jnp.maximum(gamma[jnp.newaxis, :], 1e-10)  # (n, c)
        denominator = jnp.power(ratio, exponent)  # (n, c)

        # Compute typicality with numerical stability
        T = 1.0 / (1.0 + denominator)

        # Clip to valid range [0, 1]
        T = jnp.clip(T, 0.0, 1.0)

        return T

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self,
        X: chex.Array,
        centroids: chex.Array,
        T: chex.Array,
        gamma: chex.Array
    ) -> chex.Array:
        """
        Compute PCM objective function.

        J = Σᵢ Σⱼ tᵢⱼᵐ ||xᵢ - vⱼ||² + Σⱼ γⱼ Σᵢ (1 - tᵢⱼ)ᵐ

        Args:
            X: Data matrix of shape (n, d)
            centroids: Cluster centroids of shape (c, d)
            T: Typicality matrix of shape (n, c)
            gamma: Scale parameters of shape (c,)

        Returns:
            objective: Scalar objective value
        """
        # First term: Σᵢ Σⱼ tᵢⱼᵐ ||xᵢ - vⱼ||²
        D_sq = squared_euclidean_distance_matrix(X, centroids)  # (n, c)
        T_fuzz = jnp.power(T, self.fuzzifier)  # (n, c)
        term1 = jnp.sum(T_fuzz * D_sq)

        # Second term: Σⱼ γⱼ Σᵢ (1 - tᵢⱼ)ᵐ
        one_minus_T = 1.0 - T  # (n, c)
        one_minus_T_fuzz = jnp.power(one_minus_T, self.fuzzifier)  # (n, c)
        sum_per_cluster = jnp.sum(one_minus_T_fuzz, axis=0)  # (c,)
        term2 = jnp.sum(gamma * sum_per_cluster)

        objective = term1 + term2

        return objective

    @partial(jit, static_argnums=(0,))
    def _single_iteration(self, state: PCMState, X: chex.Array) -> PCMState:
        """
        Perform a single iteration of PCM.

        Args:
            state: Current PCM state
            X: Data matrix of shape (n, d)

        Returns:
            new_state: Updated PCM state
        """
        # Update centroids
        centroids_new = self._compute_centroids(X, state.T)

        # Update gamma
        gamma_new = self._compute_gamma(X, state.T, centroids_new)

        # Update typicality
        T_new = self._update_typicality(X, centroids_new, gamma_new)

        # Compute objective
        objective_new = self._compute_objective(X, centroids_new, T_new, gamma_new)

        # Check convergence (centroid change)
        centroid_change = jnp.linalg.norm(centroids_new - state.centroids)
        converged = centroid_change < self.epsilon

        new_state = PCMState(
            centroids=centroids_new,
            T=T_new,
            gamma=gamma_new,
            objective=objective_new,
            iteration=state.iteration + 1,
            converged=converged
        )

        return new_state

    @partial(jit, static_argnums=(0,))
    def _fit_loop(self, X: chex.Array, initial_state: PCMState):
        """
        JIT-compiled training loop using jax.lax.scan.

        Args:
            X: Data matrix of shape (n, d)
            initial_state: Initial PCM state

        Returns:
            final_state: Final PCM state after training
            history: Array of objective values per iteration
        """
        def scan_fn(state, _):
            new_state = self._single_iteration(state, X)
            return new_state, new_state.objective

        # Run scan for max_iter iterations
        final_state, objectives = jax.lax.scan(
            scan_fn,
            initial_state,
            None,
            length=self.max_iter
        )

        return final_state, objectives

    def fit(self, X: chex.Array) -> 'PCM_JAX':
        """
        Fit PCM clustering model to data.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            self: Fitted model

        Raises:
            ValueError: If X has invalid shape or contains invalid values
        """
        # Validate input
        chex.assert_rank(X, 2)
        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters})"
            )

        # Initialize centroids and typicality
        if self.init_method == 'fcm':
            centroids_init, T_init = self._initialize_from_fcm(X)
        else:
            centroids_init, T_init = self._initialize_random(X)

        # Compute initial gamma
        gamma_init = self._compute_gamma(X, T_init, centroids_init)

        # Compute initial objective
        objective_init = self._compute_objective(X, centroids_init, T_init, gamma_init)

        # Create initial state
        initial_state = PCMState(
            centroids=centroids_init,
            T=T_init,
            gamma=gamma_init,
            objective=objective_init,
            iteration=0,
            converged=False
        )

        # Run training loop
        final_state, objectives = self._fit_loop(X, initial_state)

        # Store results
        self.centroids_ = final_state.centroids
        self.T_ = final_state.T
        self.gamma_ = final_state.gamma
        self.n_iter_ = int(final_state.iteration)
        self.objective_ = final_state.objective
        self._objective_history = objectives

        return self

    def predict(self, X: chex.Array) -> chex.Array:
        """
        Predict cluster labels for data.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            labels: Cluster labels of shape (n_samples,)

        Raises:
            ValueError: If model has not been fitted
        """
        if self.centroids_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        chex.assert_rank(X, 2)

        # Compute distances to centroids
        D = euclidean_distance_matrix(X, self.centroids_)  # (n, c)

        # Assign to nearest centroid
        labels = jnp.argmin(D, axis=1)

        return labels

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """
        Predict typicality values for data.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            T: Typicality matrix of shape (n_samples, n_clusters)

        Raises:
            ValueError: If model has not been fitted
        """
        if self.centroids_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        chex.assert_rank(X, 2)

        # Compute typicality using learned gamma
        T = self._update_typicality(X, self.centroids_, self.gamma_)

        return T

    def get_objective_history(self) -> chex.Array:
        """
        Get history of objective function values during training.

        Returns:
            objectives: Array of objective values, shape (n_iterations,)

        Raises:
            ValueError: If model has not been fitted
        """
        if self._objective_history is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        return self._objective_history
