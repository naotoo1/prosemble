"""
JAX implementation of K-means++ (Hard C-Means with K-means++ initialization)

This is a GPU-accelerated implementation using JAX.
K-means++ provides better initialization than random selection.
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT

from typing import Self

import jax
import jax.numpy as jnp
import chex
from jax import jit
from functools import partial

from prosemble.core.distance import batch_squared_euclidean
from .hcm import HCM


class KMeansPlusPlus:
    """
    K-means++ clustering with JAX (Hard C-Means with smart initialization)

    K-means++ is an algorithm for choosing initial cluster centers with
    better convergence properties than random initialization.

    Algorithm:
    1. Choose first center uniformly at random from data points
    2. For each data point x, compute D(x), the distance to nearest center
    3. Choose next center with probability proportional to D(x)²
    4. Repeat until k centers are chosen
    5. Run standard k-means (HCM) with these initial centers

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters
    max_iter : int, default=100
        Maximum number of iterations
    epsilon : float, default=1e-5
        Convergence tolerance
    random_seed : int, optional
        Random seed for reproducibility
    plot_steps : bool, default=False
        Whether to enable visualization (requires LiveVisualizer)

    Attributes
    ----------
    centroids_ : array of shape (n_clusters, n_features)
        Cluster centers
    labels_ : array of shape (n_samples,)
        Labels of each point
    n_iter_ : int
        Number of iterations run
    objective_ : float
        Final objective function value
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        random_seed: int | None = None,
        plot_steps: bool = False
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_seed = random_seed
        self.plot_steps = plot_steps

        # Fitted attributes
        self.centroids_ = None
        self.labels_ = None
        self.n_iter_ = 0
        self.objective_ = None
        self._hcm = None

    def _kmeans_plusplus_init(self, X: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """
        Initialize centroids using k-means++ algorithm.

        Uses a Python loop (not lax.fori_loop) because each iteration
        changes the number of selected centroids.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data
        key : PRNGKey
            Random key for JAX

        Returns
        -------
        centroids : array of shape (n_clusters, n_features)
            Initial cluster centers
        """
        n_samples, n_features = X.shape

        # Step 1: Choose first center uniformly at random
        key, subkey = jax.random.split(key)
        first_idx = jax.random.choice(subkey, n_samples)
        centroids = X[first_idx:first_idx+1]  # Shape: (1, n_features)

        # Steps 2-4: Choose remaining centers
        for _ in range(self.n_clusters - 1):
            # Compute squared distances to nearest centroid
            D_sq = batch_squared_euclidean(X, centroids)  # (n_samples, n_centers)
            min_distances = jnp.min(D_sq, axis=1)  # (n_samples,)

            # Sample proportional to squared distance
            key, subkey = jax.random.split(key)
            probs = min_distances / jnp.maximum(jnp.sum(min_distances), 1e-10)
            next_idx = jax.random.choice(subkey, n_samples, p=probs)
            new_cent = X[next_idx:next_idx+1]
            centroids = jnp.concatenate([centroids, new_cent], axis=0)

        return centroids

    def fit(self, X: chex.Array) -> Self:
        """
        Fit K-means++ model to data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Initialize centroids using k-means++
        seed = self.random_seed if self.random_seed is not None else 42
        key = jax.random.PRNGKey(seed)
        initial_centroids = self._kmeans_plusplus_init(X, key)

        # Use HCM with k-means++ initialization
        self._hcm = HCM(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            random_seed=self.random_seed,
            plot_steps=self.plot_steps
        )

        # Pass k-means++ centroids to HCM
        self._hcm.fit(X, initial_centroids=initial_centroids)

        # Copy results
        self.centroids_ = self._hcm.centroids_
        self.labels_ = self._hcm.labels_
        self.n_iter_ = self._hcm.n_iter_
        self.objective_ = self._hcm.objective_

        return self

    def predict(self, X: chex.Array) -> chex.Array:
        """
        Predict cluster labels for samples.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            New data to predict

        Returns
        -------
        labels : array of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        if self._hcm is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self._hcm.predict(X)

    def get_objective_history(self):
        """Get the objective function history."""
        if self._hcm is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self._hcm.get_objective_history()

    def final_centroids(self):
        """Get final cluster centroids."""
        if self.centroids_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.centroids_

    def get_distance_space(self, X: chex.Array) -> chex.Array:
        """
        Compute distance space (distances to all centroids).

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data

        Returns
        -------
        distances : array of shape (n_samples, n_clusters)
            Distances to each centroid
        """
        if self._hcm is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self._hcm.get_distance_space(X)


# Alias for backward compatibility
kmeans_plusplus_jax = KMeansPlusPlus
Kmeans = KMeansPlusPlus
