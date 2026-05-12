"""
JAX-based kernel functions for kernel clustering methods.

This module provides GPU-accelerated kernel computations using JAX.
"""

import jax.numpy as jnp
import chex
from jax import jit
from functools import partial


@jit
def gaussian_kernel(x: chex.Array, y: chex.Array, sigma: float) -> chex.Array:
    """
    Compute Gaussian (RBF) kernel between two vectors.

    K(x, y) = exp(-||x - y||² / (2σ²))

    Args:
        x: First vector, shape (n_features,)
        y: Second vector, shape (n_features,)
        sigma: Kernel bandwidth parameter

    Returns:
        Kernel value (scalar)
    """
    diff = x - y
    sq_norm = jnp.sum(diff * diff)
    return jnp.exp(-sq_norm / (2.0 * sigma ** 2))


@jit
def batch_gaussian_kernel(
    X: chex.Array, Y: chex.Array, sigma: float
) -> chex.Array:
    """
    Compute Gaussian kernel between two sets of vectors.

    Args:
        X: First set of vectors, shape (n_samples, n_features)
        Y: Second set of vectors, shape (m_samples, n_features)
        sigma: Kernel bandwidth parameter

    Returns:
        Kernel matrix, shape (n_samples, m_samples)
        K[i, j] = K(X[i], Y[j])
    """
    # Compute squared Euclidean distances
    X_sq = jnp.sum(X ** 2, axis=1, keepdims=True)  # (n_samples, 1)
    Y_sq = jnp.sum(Y ** 2, axis=1, keepdims=True)  # (m_samples, 1)

    # ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
    sq_distances = X_sq + Y_sq.T - 2.0 * (X @ Y.T)  # (n_samples, m_samples)

    # Ensure non-negative (numerical stability)
    sq_distances = jnp.maximum(sq_distances, 0.0)

    # Compute kernel
    K = jnp.exp(-sq_distances / (2.0 * sigma ** 2))

    return K


@jit
def kernel_distance_squared(
    X: chex.Array, Y: chex.Array, sigma: float
) -> chex.Array:
    """
    Compute squared distance in feature space: ||φ(x) - φ(y)||²

    For Gaussian kernel::

        ||φ(x) - φ(y)||² = K(x,x) + K(y,y) - 2K(x,y)
                          = 2(1 - K(x,y))  [since K(x,x) = 1]

    Args:
        X: First set of vectors, shape (n_samples, n_features)
        Y: Second set of vectors, shape (m_samples, n_features)
        sigma: Kernel bandwidth parameter

    Returns:
        Squared distances in feature space, shape (n_samples, m_samples)
    """
    K = batch_gaussian_kernel(X, Y, sigma)
    return 2.0 * (1.0 - K)


@jit
def kernel_distance(
    X: chex.Array, Y: chex.Array, sigma: float
) -> chex.Array:
    """
    Compute distance in feature space: ||φ(x) - φ(y)||

    Args:
        X: First set of vectors, shape (n_samples, n_features)
        Y: Second set of vectors, shape (m_samples, n_features)
        sigma: Kernel bandwidth parameter

    Returns:
        Distances in feature space, shape (n_samples, m_samples)
    """
    D_sq = kernel_distance_squared(X, Y, sigma)
    return jnp.sqrt(jnp.maximum(D_sq, 0.0))
