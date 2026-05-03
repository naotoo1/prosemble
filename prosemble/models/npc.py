"""
JAX implementation of Noise Possibilistic C-Means (NPC)

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


class NPC:
    """
    Noise Possibilistic C-Means (NPC) with JAX

    NPC is a supervised prototype-based classifier that iteratively optimizes
    prototypes based on accuracy metric. Uses softmin for probability estimation.

    Algorithm:
    1. Initialize prototypes (one per class)
    2. Predict labels using nearest prototype
    3. Compute accuracy
    4. If accuracy >= threshold or max_iter reached, stop
    5. Otherwise, recompute prototypes and repeat

    Softmin function: softmin(x_i) = exp(-x_i) / Σ_j exp(-x_j)

    Parameters
    ----------
    n_classes : int
        Number of classes
    max_iter : int, default=10
        Maximum optimization steps
    tol : float, default=0.8
        Accuracy threshold for convergence
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_classes: int = 3,
        max_iter: int = 10,
        tol: float = 0.8,
        random_state: int | None = None
    ):
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Fitted attributes
        self.prototypes_ = None
        self.n_iter_ = 0

    def _compute_prototypes(self, X: chex.Array, y: chex.Array) -> chex.Array:
        """
        Compute prototype for each class as the mean of samples in that class

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,)

        Returns
        -------
        prototypes : array of shape (n_classes, n_features)
        """
        # Cannot JIT this due to boolean indexing
        # Use vectorized approach instead
        prototypes = []

        for class_idx in range(self.n_classes):
            # Get samples belonging to this class
            mask = (y == class_idx)
            class_samples = X[mask]

            # Compute mean
            if jnp.sum(mask) > 0:
                prototype = jnp.mean(class_samples, axis=0)
            else:
                # If no samples, use random initialization
                prototype = jnp.zeros(X.shape[1])

            prototypes.append(prototype)

        return jnp.array(prototypes)

    @partial(jit, static_argnums=(0,))
    def _predict_labels(self, X: chex.Array, prototypes: chex.Array) -> chex.Array:
        """Predict class labels using nearest prototype"""
        D_sq = batch_squared_euclidean(X, prototypes)
        labels = jnp.argmin(D_sq, axis=1)
        return labels

    @partial(jit, static_argnums=(0,))
    def _compute_accuracy(self, y_true: chex.Array, y_pred: chex.Array) -> float:
        """Compute classification accuracy"""
        return jnp.mean(y_true == y_pred)

    def fit(self, X: chex.Array, y: chex.Array) -> Self:
        """
        Fit NPC model to labeled data

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels (integers from 0 to n_classes-1)

        Returns
        -------
        self
        """
        X = jnp.asarray(X)
        y = jnp.asarray(y)

        # Initialize prototypes
        prototypes = self._compute_prototypes(X, y)

        # Optimization loop
        for iteration in range(self.max_iter):
            # Predict
            y_pred = self._predict_labels(X, prototypes)

            # Compute accuracy
            accuracy = self._compute_accuracy(y, y_pred)

            # Check convergence
            if accuracy >= self.tol:
                self.n_iter_ = iteration + 1
                break

            # Update prototypes
            prototypes = self._compute_prototypes(X, y)

            # Store iteration count
            self.n_iter_ = iteration + 1

        self.prototypes_ = prototypes
        return self

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
        if self.prototypes_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = jnp.asarray(X)
        return self._predict_labels(X, self.prototypes_)

    @partial(jit, static_argnums=(0,))
    def _softmin(self, x: chex.Array) -> chex.Array:
        """
        Softmin function: softmin(x_i) = exp(-x_i) / Σ_j exp(-x_j)

        Parameters
        ----------
        x : array of shape (n_prototypes,)
            Distance values

        Returns
        -------
        probs : array of shape (n_prototypes,)
            Softmin probabilities
        """
        neg_x = -x
        neg_x_shifted = neg_x - jnp.max(neg_x)
        exp_neg_x = jnp.exp(neg_x_shifted)
        return exp_neg_x / jnp.sum(exp_neg_x)

    @partial(jit, static_argnums=(0,))
    def _compute_distance_space(self, X: chex.Array) -> chex.Array:
        """
        Compute distance matrix to all prototypes

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        distances : array of shape (n_samples, n_classes)
            Euclidean distances to each prototype
        """
        D_sq = batch_squared_euclidean(X, self.prototypes_)
        D = jnp.sqrt(jnp.maximum(D_sq, 1e-10))
        return D

    @partial(jit, static_argnums=(0,))
    def _predict_proba(self, X: chex.Array) -> chex.Array:
        """Compute class probabilities using softmin"""
        distances = self._compute_distance_space(X)

        # Apply softmin to each sample
        probs = jax.vmap(self._softmin)(distances)
        return probs

    def predict_proba(self, X: chex.Array) -> chex.Array:
        """
        Predict class probabilities for samples using softmin

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict

        Returns
        -------
        probs : array of shape (n_samples, n_classes)
            Class probabilities (softmin of distances)
        """
        if self.prototypes_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = jnp.asarray(X)
        return self._predict_proba(X)

    def get_distance_space(self, X: chex.Array) -> chex.Array:
        """
        Get distance space (Euclidean distances to prototypes)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to compute distances for

        Returns
        -------
        distances : array of shape (n_samples, n_classes)
            Euclidean distances to each prototype
        """
        if self.prototypes_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = jnp.asarray(X)
        return self._compute_distance_space(X)
