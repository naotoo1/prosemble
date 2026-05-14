"""
JAX-based Hard C-Means (HCM) / K-Means clustering implementation.

This module provides a GPU-accelerated implementation of Hard C-Means (K-Means)
using JAX with JIT compilation for high performance.
"""

from typing import NamedTuple, Self
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import chex
from jax import jit, lax

from prosemble.models.base import FuzzyClusteringBase, ScanFitMixin


class HCMState(NamedTuple):
    """Immutable state for HCM iteration.

    Attributes:
        centroids: Cluster centroids, shape (n_clusters, n_features)
        labels: Hard cluster assignments, shape (n_samples,)
        objective: Sum of squared distances to assigned centroids
        iteration: Current iteration number
        converged: Whether algorithm has converged
    """
    centroids: chex.Array
    labels: chex.Array
    objective: chex.Array
    iteration: int
    converged: bool


class HCM(ScanFitMixin, FuzzyClusteringBase):
    """
    Hard C-Means (K-Means) clustering with JAX.

    HCM assigns each data point to exactly one cluster (hard assignment) based on
    the nearest centroid. This is the classic K-Means algorithm.

    Algorithm:

    1. Initialize centroids randomly or from data
    2. Assign each point to nearest centroid
    3. Update centroids as mean of assigned points
    4. Repeat until convergence

    Objective function::

        J = Σ_i ||x_i - v_{label_i}||²

    Parameters
    ----------
    n_clusters : int
        Number of clusters (must be >= 2)
    max_iter : int, default=100
        Maximum number of iterations
    epsilon : float, default=1e-5
        Convergence threshold for centroid change
    init_method : {'random', 'kmeans++'}, default='random'
        Method for initializing centroids
    random_seed : int, default=42
        Random seed for reproducibility
    plot_steps : bool, default=False
        Whether to visualize clustering progress (2D PCA projection)
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
    labels_ : array, shape (n_samples,)
        Hard cluster assignments for training data
    n_iter_ : int
        Number of iterations until convergence
    objective_ : float
        Final objective function value
    objective_history_ : array
        Objective value at each iteration

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from prosemble.models import HCM
    >>> X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])
    >>> model = HCM(n_clusters=2, random_seed=42)
    >>> model.fit(X)
    >>> labels = model.predict(X)

    See Also
    --------
    FuzzyClusteringBase : Full list of base parameters (distance_fn,
        patience, restore_best, callbacks, etc.).
    """

    _hyperparams = ('init_method',)
    _fitted_array_names = ('labels_',)

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        init_method: str = 'random',
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
        if init_method not in ['random', 'kmeans++']:
            raise ValueError("init_method must be 'random' or 'kmeans++'")

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

        self.init_method = init_method

        # Model-specific fitted attributes
        self.labels_ = None

    def _initialize_centroids(self, X: chex.Array) -> chex.Array:
        """Initialize cluster centroids.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            Initial centroids, shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]

        if self.init_method == 'random':
            # Randomly select data points as initial centroids
            indices = jax.random.choice(
                self.key, n_samples, shape=(self.n_clusters,), replace=False
            )
            centroids = X[indices]
        elif self.init_method == 'kmeans++':
            # K-means++ initialization
            centroids = self._kmeans_plusplus_init(X)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        return centroids

    def _kmeans_plusplus_init(self, X: chex.Array) -> chex.Array:
        """Initialize centroids using K-means++ algorithm.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            Initial centroids, shape (n_clusters, n_features)
        """
        n_samples, n_features = X.shape
        centroids = jnp.zeros((self.n_clusters, n_features))

        # Choose first centroid uniformly at random
        key1, key2 = jax.random.split(self.key)
        first_idx = jax.random.choice(key1, n_samples)
        centroids = centroids.at[0].set(X[first_idx])

        # Choose remaining centroids with probability proportional to distance squared
        for i in range(1, self.n_clusters):
            # Compute distances to nearest existing centroid
            D_sq = self.distance_fn(X, centroids[:i])  # (n_samples, i)
            min_distances = jnp.min(D_sq, axis=1)  # (n_samples,)

            # Probability proportional to squared distance
            probs = min_distances / jnp.sum(min_distances)

            # Choose next centroid
            key2, subkey = jax.random.split(key2)
            next_idx = jax.random.choice(subkey, n_samples, p=probs)
            centroids = centroids.at[i].set(X[next_idx])

        return centroids

    @partial(jit, static_argnums=(0,))
    def _assign_labels(self, X: chex.Array, centroids: chex.Array) -> chex.Array:
        """Assign each data point to nearest centroid (hard assignment).

        Args:
            X: Input data, shape (n_samples, n_features)
            centroids: Current centroids, shape (n_clusters, n_features)

        Returns:
            labels: Hard cluster assignments, shape (n_samples,)
        """
        # Compute squared distances to all centroids
        D_sq = self.distance_fn(X, centroids)  # (n_samples, n_clusters)

        # Assign to nearest centroid
        labels = jnp.argmin(D_sq, axis=1)  # (n_samples,)

        return labels

    @partial(jit, static_argnums=(0,))
    def _update_centroids(self, X: chex.Array, labels: chex.Array) -> chex.Array:
        """Update centroids as mean of assigned points.

        Args:
            X: Input data, shape (n_samples, n_features)
            labels: Hard cluster assignments, shape (n_samples,)

        Returns:
            centroids: Updated centroids, shape (n_clusters, n_features)
        """
        n_features = X.shape[1]

        def compute_centroid(cluster_idx):
            # Get mask for points assigned to this cluster
            mask = labels == cluster_idx  # (n_samples,)

            # Count points in cluster
            count = jnp.sum(mask)

            # Compute mean of assigned points
            # If no points assigned, keep old centroid (handled by where)
            sum_points = jnp.sum(jnp.where(mask[:, None], X, 0.0), axis=0)
            centroid = jnp.where(count > 0, sum_points / count, 0.0)

            return centroid

        # Vectorize over clusters
        centroids = jax.vmap(compute_centroid)(jnp.arange(self.n_clusters))

        return centroids

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self, X: chex.Array, labels: chex.Array, centroids: chex.Array
    ) -> chex.Array:
        """Compute HCM objective function.

        Objective: J = Σ_i ||x_i - v_{label_i}||²

        Args:
            X: Input data, shape (n_samples, n_features)
            labels: Hard cluster assignments, shape (n_samples,)
            centroids: Current centroids, shape (n_clusters, n_features)

        Returns:
            objective: Scalar objective value
        """
        # Get assigned centroids for each point
        assigned_centroids = centroids[labels]  # (n_samples, n_features)

        # Compute squared distances
        diff = X - assigned_centroids
        sq_distances = jnp.sum(diff * diff, axis=1)  # (n_samples,)

        # Sum over all points
        objective = jnp.sum(sq_distances)

        return objective

    @partial(jit, static_argnums=(0,))
    def _iteration_step(
        self, state: HCMState, X: chex.Array
    ) -> tuple[HCMState, dict]:
        """Single HCM iteration step.

        Args:
            state: Current HCM state
            X: Input data, shape (n_samples, n_features)

        Returns:
            new_state: Updated HCM state
            metrics: Dictionary of metrics for this iteration
        """
        # Assign labels based on current centroids
        labels = self._assign_labels(X, state.centroids)

        # Update centroids based on new assignments
        new_centroids = self._update_centroids(X, labels)

        # Compute objective
        objective = self._compute_objective(X, labels, new_centroids)

        # Check convergence based on centroid change
        centroid_change = jnp.linalg.norm(new_centroids - state.centroids, ord='fro')
        converged = centroid_change <= self.epsilon

        new_state = HCMState(
            centroids=new_centroids,
            labels=labels,
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
        weights = jnp.ones(state.labels.shape[0])
        return {
            'centroids': state.centroids, 'labels': state.labels,
            'weights': weights, 'iteration': iteration,
            'objective': float(state.objective), 'max_iter': self.max_iter,
        }

    def fit(self, X: chex.Array, initial_centroids=None, resume=False) -> Self:
        """Fit HCM model to data.

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

        # Initialize centroids
        if resume:
            self._check_fitted()
            centroids_init = self.centroids_
        elif initial_centroids is not None:
            centroids_init = self._validate_initial_centroids(X, initial_centroids)
        else:
            centroids_init = self._initialize_centroids(X)

        initial_labels = self._assign_labels(X, centroids_init)
        initial_objective = self._compute_objective(X, initial_labels, centroids_init)
        initial_state = HCMState(
            centroids=centroids_init, labels=initial_labels,
            objective=initial_objective, iteration=0, converged=False
        )

        final_state, self.history_ = self._run_training(X, initial_state)

        # Store results
        self.centroids_ = final_state.centroids
        self.labels_ = final_state.labels
        self.n_iter_ = int(final_state.iteration)
        self.objective_ = float(final_state.objective)
        self.objective_history_ = self.history_['objective']

        return self

    def predict(self, X: chex.Array) -> chex.Array:
        """Predict cluster labels for new data.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            labels: Cluster labels, shape (n_samples,)

        Raises:
            ValueError: If model has not been fitted
        """
        self._check_fitted()

        X = jnp.asarray(X)

        labels = self._assign_labels(X, self.centroids_)
        return labels

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """Predict hard cluster membership (one-hot encoding).

        For HCM, this returns a one-hot encoding of the cluster assignments,
        with 1.0 for the assigned cluster and 0.0 for others.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            membership: One-hot encoded assignments, shape (n_samples, n_clusters)

        Raises:
            ValueError: If model has not been fitted
        """
        self._check_fitted()

        labels = self.predict(X)
        n_samples = X.shape[0]

        # Create one-hot encoding
        membership = jnp.zeros((n_samples, self.n_clusters))
        membership = membership.at[jnp.arange(n_samples), labels].set(1.0)

        return membership

    def get_distance_space(self, X: chex.Array) -> chex.Array:
        """Compute distances to all cluster centroids.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            distances: Euclidean distances to centroids, shape (n_samples, n_clusters)

        Raises:
            ValueError: If model has not been fitted
        """
        self._check_fitted()

        X = jnp.asarray(X)

        # Compute squared distances
        D_sq = self.distance_fn(X, self.centroids_)

        # Return Euclidean distances
        distances = jnp.sqrt(D_sq)

        return distances
