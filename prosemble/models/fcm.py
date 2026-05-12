"""
JAX implementation of Fuzzy C-Means (FCM) clustering algorithm.

This module provides a GPU-accelerated, fully vectorized implementation of FCM
using JAX. All operations are JIT-compiled for maximum performance.

Mathematical Background:
-----------------------
FCM is a soft clustering algorithm that assigns membership degrees to data points
for each cluster. Unlike hard clustering (K-Means), each point can belong to
multiple clusters with varying degrees.

Objective Function:
    J(U, V) = Σᵢ Σⱼ uᵢⱼᵐ ||xᵢ - vⱼ||²

Subject to:
    Σⱼ uᵢⱼ = 1  ∀i  (membership constraint)
    uᵢⱼ ∈ [0,1]    (fuzzy membership)

Update Rules:
    vⱼ = Σᵢ(uᵢⱼᵐ xᵢ) / Σᵢ uᵢⱼᵐ
    uᵢⱼ = 1 / Σₖ (dᵢⱼ/dᵢₖ)^(2/(m-1))

where:
    - U: fuzzy membership matrix (n × c)
    - V: cluster centroids (c × d)
    - m: fuzzifier parameter (typically 2.0)
    - dᵢⱼ: distance from point xᵢ to centroid vⱼ

Author: Prosemble Contributors
License: MIT
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from functools import partial
from typing import NamedTuple, Self
import chex

from prosemble.models.base import FuzzyClusteringBase, ScanFitMixin


class FCMState(NamedTuple):
    """
    Immutable state for FCM algorithm.

    JAX requires immutable state for JIT compilation and functional programming.
    Using NamedTuple ensures state cannot be modified in-place.

    Attributes:
        centroids: (c, d) array of cluster centroids
        U: (n, c) array of fuzzy membership values
        objective: Scalar objective function value
        iteration: Current iteration number
        converged: Boolean indicating convergence
    """
    centroids: chex.Array
    U: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class FCM(ScanFitMixin, FuzzyClusteringBase):
    """
    JAX implementation of Fuzzy C-Means clustering.

    This implementation provides:
    - Full vectorization (no Python loops)
    - JIT compilation for speed
    - Automatic GPU acceleration
    - Immutable state management
    - Numerical stability

    Key Differences from NumPy Version:
    -----------------------------------
    1. **Vectorization**: All operations use matrix operations
       Old: Triple nested loops in centroid computation
       New: Single matrix multiplication

    2. **Functional**: Immutable state using NamedTuple
       Old: In-place updates (self.fit_cent = ...)
       New: Return new state objects

    3. **JIT Compilation**: Functions compiled to machine code
       Old: Interpreted Python loops
       New: Compiled XLA code

    4. **GPU Support**: Automatic device placement
       Old: CPU-only NumPy
       New: GPU/TPU with JAX

    Parameters
    ----------
    n_clusters : int
        Number of clusters (c)

    fuzzifier : float, default=2.0
        Fuzzification parameter (m). Must be > 1.
        - m = 1: Hard clustering (crisp membership)
        - m = 2: Standard fuzzy clustering
        - m → ∞: Maximum fuzziness (equal membership)

    max_iter : int, default=100
        Maximum number of iterations

    epsilon : float, default=1e-5
        Convergence tolerance. Algorithm stops when:
        ||V_new - V_old||_F < epsilon

    init_method : str, default='random'
        Initialization method for U matrix:
        - 'random': Random Dirichlet distribution
        - 'kmeans++': K-means++ centroids then compute U

    random_seed : int, default=42
        Random seed for reproducibility

    Attributes
    ----------
    centroids_ : array of shape (n_clusters, n_features)
        Cluster centroids after fitting

    U_ : array of shape (n_samples, n_clusters)
        Fuzzy membership matrix after fitting

    objective_ : float
        Final objective function value

    n_iter_ : int
        Number of iterations performed

    history_ : dict
        Training history containing objective values and other metrics

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from prosemble.models import FCM
    >>>
    >>> # Generate sample data
    >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    >>>
    >>> # Fit FCM model
    >>> model = FCM(n_clusters=2, fuzzifier=2.0, max_iter=100)
    >>> model.fit(X)
    >>>
    >>> # Get results
    >>> labels = model.predict(X)
    >>> centroids = model.final_centroids()
    >>> membership = model.predict_proba(X)
    >>>
    >>> print(f"Labels: {labels}")
    >>> print(f"Centroids shape: {centroids.shape}")
    >>> print(f"Membership shape: {membership.shape}")

    References
    ----------
    Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective
    Function Algorithms. Plenum Press, New York.

    Dunn, J. C. (1973). A Fuzzy Relative of the ISODATA Process and
    Its Use in Detecting Compact Well-Separated Clusters.

    See Also
    --------
    FuzzyClusteringBase : Full list of base parameters (distance_fn,
        patience, restore_best, callbacks, etc.).
    """

    _hyperparams = ('fuzzifier', 'init_method')
    _fitted_array_names = ('U_',)

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        init_method: str = 'random',
        random_seed: int = 42,
        distance_fn=None,
        plot_steps: bool = False,
        show_confidence: bool = True,
        show_pca_variance: bool = True,
        save_plot_path: str = None,
        **kwargs
    ):
        # Model-specific validation first
        if fuzzifier <= 1.0:
            raise ValueError(f"fuzzifier must be > 1, got {fuzzifier}")

        super().__init__(
            n_clusters=n_clusters, max_iter=max_iter, epsilon=epsilon,
            random_seed=random_seed, distance_fn=distance_fn, plot_steps=plot_steps,
            show_confidence=show_confidence, show_pca_variance=show_pca_variance,
            save_plot_path=save_plot_path, **kwargs
        )

        self.fuzzifier = fuzzifier
        self.init_method = init_method

        # Model-specific fitted attributes
        self.U_ = None
        self.history_ = None

    @partial(jit, static_argnums=(0,))
    def _initialize_U(self, X: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """
        Initialize fuzzy membership matrix U.

        Uses Dirichlet distribution to ensure row sums equal 1:
        U ~ Dir(α) where α = [1, 1, ..., 1]

        Mathematical Property:
            Σⱼ uᵢⱼ = 1  ∀i

        Args:
            X: (n, d) data matrix
            key: JAX random key for reproducibility

        Returns:
            U: (n, c) fuzzy membership matrix
        """
        n_samples = X.shape[0]

        # Dirichlet distribution ensures row sums = 1
        U = jax.random.dirichlet(
            key,
            alpha=jnp.ones(self.n_clusters),
            shape=(n_samples,)
        )

        return U

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(self, X: chex.Array, U: chex.Array) -> chex.Array:
        """
        Compute cluster centroids from fuzzy membership matrix.

        Mathematical Formula:
            vⱼ = Σᵢ(uᵢⱼᵐ xᵢ) / Σᵢ uᵢⱼᵐ

        Vectorized Implementation:
            V = (Uᵐ)ᵀ @ X / sum((Uᵐ)ᵀ, axis=1, keepdims=True)

        Old Implementation (NumPy):
            ```python
            fuzzified_assignments = [
                np.power([u_ik[i] for _, u_ik in enumerate(fuzzy_matrix)], m)
                for i in range(c)
            ]
            sum_fuzzified = [np.sum(i) for i in fuzzified_assignments]
            centroid_numerator = [
                [np.multiply(fuzzified[cluster][index], sample)
                 for index, sample in enumerate(data)]
                for cluster in range(c)
            ]
            centroid = [np.sum(v, axis=0) / sum_fuzzified[i]
                       for i, v in enumerate(centroid_numerator)]
            ```
            Complexity: O(ncd) with 3 nested loops

        New Implementation (JAX):
            ```python
            U_fuzz = jnp.power(U, m)
            numerator = U_fuzz.T @ X
            denominator = jnp.sum(U_fuzz, axis=0, keepdims=True).T
            centroids = numerator / denominator
            ```
            Complexity: O(ncd) with single matrix multiply

        Speedup: ~10-50× due to:
        - No loop overhead
        - BLAS optimized matrix multiply
        - SIMD vectorization
        - GPU parallelization

        Args:
            X: (n, d) data matrix
            U: (n, c) fuzzy membership matrix

        Returns:
            V: (c, d) centroid matrix
        """
        # Fuzzify membership: U^m
        U_fuzz = jnp.power(U, self.fuzzifier)  # (n, c)

        # Numerator: (c, n) @ (n, d) = (c, d)
        numerator = U_fuzz.T @ X

        # Denominator: (c, 1)
        denominator = jnp.sum(U_fuzz, axis=0, keepdims=True).T

        # Avoid division by zero
        denominator = jnp.maximum(denominator, 1e-10)

        centroids = numerator / denominator

        return centroids

    @partial(jit, static_argnums=(0,))
    def _update_U(
        self,
        X: chex.Array,
        centroids: chex.Array
    ) -> chex.Array:
        """
        Update fuzzy membership matrix.

        Mathematical Formula:
            uᵢⱼ = 1 / Σₖ (dᵢⱼ/dᵢₖ)^(2/(m-1))

        where dᵢⱼ = ||xᵢ - vⱼ|| is Euclidean distance.

        Old Implementation (NumPy):
            ```python
            for i in range(len(data)):
                denominator = 0
                for j in range(c):
                    denominator += np.power(
                        1 / euclidean_distance(centroids[j], data[i]),
                        2 / (m - 1)
                    )
                for j in range(c):
                    uik_new = np.power(
                        1 / euclidean_distance(centroids[j], data[i]),
                        2 / (m - 1)
                    ) / denominator
                    u_matrix[i][j] = uik_new
            ```
            Issues:
            - Double nested loop
            - Distance computed twice
            - In-place updates

        New Implementation (JAX):
            ```python
            D = euclidean_distance_matrix(X, centroids)  # (n, c)
            D = jnp.maximum(D, 1e-10)  # Numerical stability
            power = 2.0 / (m - 1)
            ratios = (D[:, :, None] / D[:, None, :]) ** power  # (n, c, c)
            denominators = jnp.sum(ratios, axis=2)  # (n, c)
            U = 1.0 / denominators
            ```
            Benefits:
            - Single distance computation
            - Vectorized operations
            - Immutable (functional)

        Args:
            X: (n, d) data matrix
            centroids: (c, d) centroid matrix

        Returns:
            U: (n, c) updated fuzzy membership matrix
        """
        # Compute pairwise distances: (n, c)
        D = self.distance_fn(X, centroids)

        # Add small epsilon to avoid division by zero
        D = jnp.maximum(D, 1e-10)

        # Compute power for formula
        power = 1.0 / (self.fuzzifier - 1)

        # For each i, j: sum over k of (d_ij / d_ik)^power
        # Reshape for broadcasting: (n, c, 1) / (n, 1, c) = (n, c, c)
        ratios = jnp.power(D[:, :, None] / D[:, None, :], power)

        # Sum over k dimension: (n, c, c) -> (n, c)
        denominators = jnp.sum(ratios, axis=2)

        # U_ij = 1 / denominator_ij
        U = 1.0 / denominators

        # Normalize rows to sum to 1 (numerical stability)
        U = U / jnp.sum(U, axis=1, keepdims=True)

        return U

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self,
        X: chex.Array,
        centroids: chex.Array,
        U: chex.Array
    ) -> chex.Array:
        """
        Compute FCM objective function.

        Mathematical Formula:
            J = Σᵢ Σⱼ uᵢⱼᵐ ||xᵢ - vⱼ||²

        Vectorized Implementation:
            J = sum(Uᵐ ⊙ D²)

        where ⊙ is element-wise product.

        Old Implementation (NumPy):
            ```python
            objective = np.sum([
                [squared_euclidean_distance(data[i], centroids[j]) *
                 np.power(u_matrix[i][j], m)
                 for i in range(len(data))]
                for j in range(c)
            ])
            ```
            Issues:
            - Nested loops
            - Redundant distance computation

        New Implementation (JAX):
            ```python
            D_sq = squared_euclidean_distance_matrix(X, centroids)
            U_fuzz = jnp.power(U, m)
            objective = jnp.sum(U_fuzz * D_sq)
            ```
            Benefits:
            - Single pass
            - Reuses distance matrix
            - Element-wise operations

        Args:
            X: (n, d) data matrix
            centroids: (c, d) centroids
            U: (n, c) fuzzy membership

        Returns:
            J: scalar objective value
        """
        # Squared distances: (n, c)
        D_sq = self.distance_fn(X, centroids)

        # Fuzzified membership: (n, c)
        U_fuzz = jnp.power(U, self.fuzzifier)

        # Element-wise multiply and sum all
        objective = jnp.sum(U_fuzz * D_sq)

        return objective

    @partial(jit, static_argnums=(0,))
    def _check_convergence(
        self,
        centroids_old: chex.Array,
        centroids_new: chex.Array
    ) -> chex.Array:
        """
        Check if centroids have converged.

        Formula: ||V_new - V_old||_F < epsilon

        Uses Frobenius norm for stability.

        Args:
            centroids_old: Previous centroids
            centroids_new: Current centroids

        Returns:
            Boolean scalar (as JAX array)
        """
        diff = jnp.linalg.norm(centroids_new - centroids_old, ord='fro')
        return diff < self.epsilon

    @partial(jit, static_argnums=(0,))
    def _iteration_step(
        self,
        state: FCMState,
        X: chex.Array
    ) -> tuple[FCMState, dict]:
        """
        Single iteration of FCM algorithm.

        This function is JIT-compiled and used in lax.scan for fast looping.

        Algorithm:

        1. Update U matrix given current centroids
        2. Compute new centroids given updated U
        3. Compute objective function
        4. Check convergence
        5. Return new state

        Args:
            state: Current algorithm state
            X: Data matrix (passed as auxiliary data)

        Returns:
            new_state: Updated state
            metrics: Dictionary of metrics for this iteration
        """
        # Update membership matrix
        U_new = self._update_U(X, state.centroids)

        # Compute new centroids
        centroids_new = self._compute_centroids(X, U_new)

        # Compute objective
        obj_new = self._compute_objective(X, centroids_new, U_new)

        # Check convergence
        converged = self._check_convergence(state.centroids, centroids_new)

        # Create new state
        new_state = FCMState(
            centroids=centroids_new,
            U=U_new,
            objective=obj_new,
            iteration=state.iteration + 1,
            converged=converged
        )

        # Return metrics for tracking
        metrics = {
            'objective': obj_new,
            'centroid_change': jnp.linalg.norm(centroids_new - state.centroids),
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

    def fit(self, X: jnp.ndarray, initial_centroids=None, resume=False) -> Self:
        """Fit FCM model to data.

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

        if not jnp.all(jnp.isfinite(X)):
            raise ValueError("X contains NaN or Inf values")

        # Initialize
        if resume:
            self._check_fitted()
            centroids_init = self.centroids_
            U_init = self._update_U(X, centroids_init)
        elif initial_centroids is not None:
            centroids_init = self._validate_initial_centroids(X, initial_centroids)
            U_init = self._update_U(X, centroids_init)
        else:
            self.key, subkey = jax.random.split(self.key)
            U_init = self._initialize_U(X, subkey)
            centroids_init = self._compute_centroids(X, U_init)
        obj_init = self._compute_objective(X, centroids_init, U_init)

        initial_state = FCMState(
            centroids=centroids_init, U=U_init,
            objective=obj_init, iteration=0, converged=False
        )

        final_state, self.history_ = self._run_training(X, initial_state)

        # Store results
        self.centroids_ = final_state.centroids
        self.U_ = final_state.U
        self.objective_ = final_state.objective
        self.n_iter_ = final_state.iteration

        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict cluster labels for X.

        Assigns each sample to the cluster with highest membership.

        Args:
            X: (n_samples, n_features) data

        Returns:
            labels: (n_samples,) cluster assignments (0 to n_clusters-1)

        Raises:
            RuntimeError: If model not fitted yet
        """
        self._check_fitted()

        X = jnp.asarray(X)

        # Compute membership matrix
        U = self._update_U(X, self.centroids_)

        # Hard assignment: argmax over clusters
        labels = jnp.argmax(U, axis=1)

        return labels

    def predict_proba(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict fuzzy membership for X.

        Args:
            X: (n_samples, n_features) data

        Returns:
            U: (n_samples, n_clusters) fuzzy membership matrix
               Each row sums to 1, values in [0, 1]

        Raises:
            RuntimeError: If model not fitted yet
        """
        self._check_fitted()

        X = jnp.asarray(X)
        return self._update_U(X, self.centroids_)

    def get_objective_history(self) -> jnp.ndarray:
        """
        Return objective function values across iterations.

        Returns:
            objectives: (max_iter,) array of objective values

        Raises:
            RuntimeError: If model not fitted yet
        """
        if self.history_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        return self.history_['objective']

    def get_distance_space(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute distances from samples to cluster centroids.

        Args:
            X: (n_samples, n_features) data

        Returns:
            D: (n_samples, n_clusters) distance matrix

        Raises:
            RuntimeError: If model not fitted yet
        """
        self._check_fitted()

        X = jnp.asarray(X)
        return self.distance_fn(X, self.centroids_)
