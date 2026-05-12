"""
JAX implementation of K-Nearest Neighbors (KNN)

This is a GPU-accelerated implementation using JAX.
"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT

from typing import Self
from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax import jit

from prosemble.core.distance import batch_squared_euclidean


class KNN:
    """
    K-Nearest Neighbors (KNN) with JAX

    KNN is a simple, non-parametric classifier that predicts based on the
    k nearest training samples.

    Algorithm:

    1. For each test sample, compute distances to all training samples,
       find k nearest neighbors, predict label as mode (most common)
       of k neighbors' labels, and compute probability as frequency
       of predicted label.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

        # Fitted attributes
        self.X_train_ = None
        self.y_train_ = None
        self.n_classes_ = None

    def fit(self, X: chex.Array, y: chex.Array) -> Self:
        """
        Fit KNN model (store training data)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        self
        """
        X = jnp.asarray(X)
        y = jnp.asarray(y)

        self.X_train_ = X
        self.y_train_ = y
        self.n_classes_ = len(jnp.unique(y))

        return self

    @partial(jit, static_argnums=(0,))
    def _get_mode(self, labels: chex.Array) -> tuple[int, float]:
        """
        Get mode (most common label) and its probability

        Parameters
        ----------
        labels : array of shape (k,)
            Labels of k nearest neighbors

        Returns
        -------
        mode_label : int
            Most common label
        probability : float
            Frequency of mode label (count / k)
        """
        # Count occurrences of each label
        # Since we don't know n_classes at JIT time, we'll use a different approach

        # Get unique labels and their counts
        unique_labels, counts = jnp.unique(labels, return_counts=True, size=self.n_neighbors)

        # Find label with maximum count
        max_count_idx = jnp.argmax(counts)
        mode_label = unique_labels[max_count_idx]
        max_count = counts[max_count_idx]

        # Compute probability
        probability = max_count / labels.shape[0]

        return mode_label, probability

    @partial(jit, static_argnums=(0,))
    def _predict_one(self, x: chex.Array) -> tuple[int, float]:
        """
        Predict label and probability for a single sample

        Parameters
        ----------
        x : array of shape (n_features,)
            Test sample

        Returns
        -------
        label : int
            Predicted label
        probability : float
            Confidence of prediction
        """
        # Compute distances to all training samples
        x_expanded = x[None, :]  # (1, n_features)
        D_sq = batch_squared_euclidean(x_expanded, self.X_train_)  # (1, n_train)
        D_sq = D_sq.squeeze(0)  # (n_train,)

        # Find k nearest neighbors
        k_nearest_indices = jnp.argsort(D_sq)[:self.n_neighbors]

        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train_[k_nearest_indices]

        # Get mode and probability
        label, probability = self._get_mode(k_nearest_labels)

        return label, probability

    @partial(jit, static_argnums=(0,))
    def _predict_labels(self, X: chex.Array) -> chex.Array:
        """Predict labels for multiple samples"""
        # Vectorize over samples
        predict_fn = lambda x: self._predict_one(x)[0]
        labels = jax.vmap(predict_fn)(X)
        return labels

    def predict(self, X: chex.Array) -> chex.Array:
        """
        Predict class labels for samples

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict

        Returns
        -------
        labels : array of shape (n_samples,)
            Predicted class labels
        """
        if self.X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = jnp.asarray(X)
        return self._predict_labels(X)

    @partial(jit, static_argnums=(0,))
    def _get_probabilities(self, X: chex.Array) -> chex.Array:
        """Get prediction probabilities for multiple samples"""
        # Vectorize over samples
        predict_fn = lambda x: self._predict_one(x)[1]
        probabilities = jax.vmap(predict_fn)(X)
        return probabilities

    def get_proba(self, X: chex.Array) -> chex.Array:
        """
        Get prediction probabilities (frequency of predicted label among k neighbors)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict

        Returns
        -------
        probabilities : array of shape (n_samples,)
            Confidence values (frequency of mode label)
        """
        if self.X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = jnp.asarray(X)
        return self._get_probabilities(X)

    @partial(jit, static_argnums=(0,))
    def _get_distance_space(self, X: chex.Array) -> chex.Array:
        """
        Get sorted distances to training samples

        For each test sample, return sorted distances to all training samples
        """
        # Compute all distances
        D_sq = batch_squared_euclidean(X, self.X_train_)  # (n_test, n_train)
        D = jnp.sqrt(jnp.maximum(D_sq, 1e-10))

        # Sort distances for each sample
        D_sorted = jnp.sort(D, axis=1)

        return D_sorted

    def distance_space(self, X: chex.Array) -> chex.Array:
        """
        Get distance space (sorted distances to all training samples)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to compute distances for

        Returns
        -------
        distances : array of shape (n_samples, n_train_samples)
            Sorted Euclidean distances to training samples
        """
        if self.X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = jnp.asarray(X)
        return self._get_distance_space(X)

    @partial(jit, static_argnums=(0,))
    def _predict_proba_full(self, X: chex.Array) -> chex.Array:
        """
        Predict class probabilities for all classes

        For each sample, return probability distribution over all classes
        based on k nearest neighbors.
        """
        def compute_proba_one(x):
            # Compute distances
            x_expanded = x[None, :]
            D_sq = batch_squared_euclidean(x_expanded, self.X_train_)
            D_sq = D_sq.squeeze(0)

            # Find k nearest neighbors
            k_nearest_indices = jnp.argsort(D_sq)[:self.n_neighbors]
            k_nearest_labels = self.y_train_[k_nearest_indices]

            # Count votes for each class
            proba = jnp.zeros(self.n_classes_)

            for class_idx in range(self.n_classes_):
                count = jnp.sum(k_nearest_labels == class_idx)
                proba = proba.at[class_idx].set(count / self.n_neighbors)

            return proba

        probas = jax.vmap(compute_proba_one)(X)
        return probas

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """
        Predict class probabilities for samples

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict

        Returns
        -------
        probabilities : array of shape (n_samples, n_classes)
            Class probability distributions
        """
        if self.X_train_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = jnp.asarray(X)
        return self._predict_proba_full(X)
