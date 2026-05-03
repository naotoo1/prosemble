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

from typing import NamedTuple, Self
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
import chex

from .fcm import FCM
from prosemble.models.base import FuzzyClusteringBase, ScanFitMixin


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


class PCM(ScanFitMixin, FuzzyClusteringBase):
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
        >>> from prosemble.models import PCM
        >>>
        >>> # Generate sample data
        >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
        >>>
        >>> # Fit PCM model
        >>> model = PCM(n_clusters=2, fuzzifier=2.0, k=1.0)
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

    _hyperparams = ('fuzzifier', 'k', 'init_method')
    _fitted_array_names = ('T_', 'gamma_')

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        k: float = 1.0,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        init_method: str = 'fcm',
        random_seed: int | None = None,
        distance_fn=None,
        plot_steps: bool = False,
        show_confidence: bool = True,
        show_pca_variance: bool = True,
        save_plot_path: str = None,
        **kwargs
    ):
        # Validate model-specific parameters
        if fuzzifier <= 1.0:
            raise ValueError(f"fuzzifier must be > 1.0, got {fuzzifier}")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if init_method not in ['fcm', 'random']:
            raise ValueError(f"init_method must be 'fcm' or 'random', got {init_method}")

        # Resolve default seed
        if random_seed is None:
            random_seed = 42

        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            epsilon=epsilon,
            random_seed=random_seed,
            distance_fn=distance_fn,
            plot_steps=plot_steps,
            show_confidence=show_confidence,
            show_pca_variance=show_pca_variance,
            save_plot_path=save_plot_path, **kwargs
        )

        self.fuzzifier = fuzzifier
        self.k = k
        self.init_method = init_method

        # Fitted attributes
        self.T_ = None
        self.gamma_ = None
        self._objective_history = None

    def _initialize_from_fcm(self, X: chex.Array) -> tuple[chex.Array, chex.Array]:
        """
        Initialize centroids and typicality matrix using FCM.

        Args:
            X: Data matrix of shape (n, d)

        Returns:
            centroids: Initial centroids of shape (c, d)
            T: Initial typicality matrix of shape (n, c)
        """
        # Run FCM to get initial centroids and membership matrix
        fcm = FCM(
            n_clusters=self.n_clusters,
            fuzzifier=self.fuzzifier,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            random_seed=int(self.key[0]),
            distance_fn=self.distance_fn,
        )
        fcm.fit(X)

        # Use FCM centroids and membership as initial typicality
        centroids = fcm.centroids_
        T = fcm.U_  # Use membership as initial typicality

        return centroids, T

    def _initialize_random(self, X: chex.Array) -> tuple[chex.Array, chex.Array]:
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
        D_sq = self.distance_fn(X, centroids)

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
        D_sq = self.distance_fn(X, centroids)

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
        D_sq = self.distance_fn(X, centroids)  # (n, c)
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
    def _iteration_step(self, state: PCMState, X: chex.Array) -> tuple[PCMState, dict]:
        """
        Perform a single iteration of PCM.

        Args:
            state: Current PCM state
            X: Data matrix of shape (n, d)

        Returns:
            new_state: Updated PCM state
            metrics: Dictionary of iteration metrics
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

        metrics = {
            'objective': new_state.objective,
            'centroid_change': centroid_change,
            'converged': new_state.converged,
        }

        return new_state, metrics

    def _build_info(self, state, iteration):
        import numpy as np
        labels = jnp.argmax(state.T, axis=1)
        weights = jnp.max(state.T, axis=1)
        max_typicality = np.asarray(weights)
        outlier_threshold = 0.5
        n_outliers = int(np.sum(max_typicality < outlier_threshold))
        return {
            'centroids': state.centroids, 'labels': labels,
            'weights': weights, 'iteration': iteration,
            'objective': float(state.objective), 'max_iter': self.max_iter,
            'outlier_count': n_outliers,
        }

    def fit(self, X: chex.Array, initial_centroids=None, resume=False) -> Self:
        """Fit PCM clustering model to data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        initial_centroids : array-like, shape (n_clusters, n_features), optional
            Pre-computed centroids for warm starting
        resume : bool, default=False
            If True, resume from the model's current fitted state
        """
        X = self._validate_input(X)

        if resume and initial_centroids is not None:
            raise ValueError("Cannot use both resume=True and initial_centroids")

        # Initialize
        if resume:
            self._check_fitted()
            centroids_init = self.centroids_
            T_init = self.T_
            gamma_init = self.gamma_
        elif initial_centroids is not None:
            centroids_init = self._validate_initial_centroids(X, initial_centroids)
            T_init = jnp.ones((X.shape[0], self.n_clusters)) / self.n_clusters
            gamma_init = self._compute_gamma(X, T_init, centroids_init)
            T_init = self._update_typicality(X, centroids_init, gamma_init)
        else:
            if self.init_method == 'fcm':
                centroids_init, T_init = self._initialize_from_fcm(X)
            else:
                centroids_init, T_init = self._initialize_random(X)

        if not resume:
            gamma_init = self._compute_gamma(X, T_init, centroids_init)
        objective_init = self._compute_objective(X, centroids_init, T_init, gamma_init)
        initial_state = PCMState(
            centroids=centroids_init, T=T_init, gamma=gamma_init,
            objective=objective_init, iteration=0, converged=False
        )

        final_state, self.history_ = self._run_training(X, initial_state)

        # Store results
        self.centroids_ = final_state.centroids
        self.T_ = final_state.T
        self.gamma_ = final_state.gamma
        self.n_iter_ = int(final_state.iteration)
        self.objective_ = final_state.objective
        self.objective_history_ = self.history_['objective']

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
        self._check_fitted()

        chex.assert_rank(X, 2)

        # Compute distances to centroids
        D = self.distance_fn(X, self.centroids_)  # (n, c)

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
        self._check_fitted()

        chex.assert_rank(X, 2)

        # Compute typicality using learned gamma
        T = self._update_typicality(X, self.centroids_, self.gamma_)

        return T
