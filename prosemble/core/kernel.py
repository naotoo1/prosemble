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

    :math:`K(x, y) = \exp(-\|x - y\|^2 / (2\sigma^2))`

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

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
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
    Compute squared distance in feature space.

    For Gaussian kernel: :math:`\|\phi(x) - \phi(y)\|^2 = K(x,x) + K(y,y) - 2K(x,y) = 2(1 - K(x,y))`,
    since :math:`K(x,x) = 1`.

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
    Compute distance in feature space.

    Args:
        X: First set of vectors, shape (n_samples, n_features)
        Y: Second set of vectors, shape (m_samples, n_features)
        sigma: Kernel bandwidth parameter

    Returns:
        Distances in feature space, shape (n_samples, m_samples)
    """
    D_sq = kernel_distance_squared(X, Y, sigma)
    return jnp.sqrt(jnp.maximum(D_sq, 0.0))


@jit
def kernel_distance_squared_per_proto(
    X: chex.Array, W: chex.Array, sigmas: chex.Array
) -> chex.Array:
    """
    Squared kernel distance with per-prototype bandwidth.

    .. math::

        d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(-\\frac{\\|x - w_k\\|^2}{2\\sigma_k^2}\\right)\\right)

    Each prototype :math:`w_k` has its own bandwidth :math:`\\sigma_k`.

    Args:
        X: Data matrix, shape (n_samples, n_features).
        W: Prototype matrix, shape (n_prototypes, n_features).
        sigmas: Per-prototype bandwidths, shape (n_prototypes,).

    Returns:
        Squared distances in feature space, shape (n_samples, n_prototypes).

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.
    """
    diff = X[:, None, :] - W[None, :, :]          # (n, p, d)
    sq_norms = jnp.sum(diff ** 2, axis=2)          # (n, p)
    K = jnp.exp(-sq_norms / (2.0 * sigmas[None, :] ** 2))  # (n, p)
    return 2.0 * (1.0 - K)


@jit
def kernel_distance_squared_relevance(
    X: chex.Array, W: chex.Array, sigmas: chex.Array,
    relevances: chex.Array
) -> chex.Array:
    """
    Squared kernel distance with relevance weighting and per-prototype bandwidth.

    .. math::

        d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(-\\frac{\\sum_j \\lambda_j (x_j - w_{kj})^2}{2\\sigma_k^2}\\right)\\right)

    Args:
        X: Data matrix, shape (n_samples, n_features).
        W: Prototype matrix, shape (n_prototypes, n_features).
        sigmas: Per-prototype bandwidths, shape (n_prototypes,).
        relevances: Normalized relevance weights, shape (n_features,).

    Returns:
        Squared distances in feature space, shape (n_samples, n_prototypes).

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.
    """
    diff = X[:, None, :] - W[None, :, :]                              # (n, p, d)
    weighted_sq = jnp.sum(relevances[None, None, :] * diff ** 2, axis=2)  # (n, p)
    K = jnp.exp(-weighted_sq / (2.0 * sigmas[None, :] ** 2))          # (n, p)
    return 2.0 * (1.0 - K)


@jit
def exponential_kernel_distance_squared(
    X: chex.Array, W: chex.Array, omega_hat: chex.Array
) -> chex.Array:
    """
    Squared distance in exponential kernel feature space.

    Uses the exponential kernel :math:`\\kappa_{\\exp}(v, w, \\hat\\Lambda) = \\exp(v^T \\hat\\Lambda w)`
    where :math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T`.

    .. math::

        d_\\kappa^2(x, w) = \\exp(x^T \\hat\\Lambda x)
                          + \\exp(w^T \\hat\\Lambda w)
                          - 2 \\exp(x^T \\hat\\Lambda w)

    Note: :math:`\\kappa(v, v) \\neq 1` for the exponential kernel, so the
    full three-term formula is required (not the 2(1-K) simplification).

    Args:
        X: Data matrix, shape (n_samples, n_features).
        W: Prototype matrix, shape (n_prototypes, n_features).
        omega_hat: Transformation matrix, shape (n_features, latent_dim).
            The kernel matrix is :math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T`.

    Returns:
        Squared distances in feature space, shape (n_samples, n_prototypes).

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.
    """
    # Λ̂ = Ω̂ Ω̂^T  (d, d)
    lambda_hat = jnp.dot(omega_hat, omega_hat.T)

    # x^T Λ̂ x for all samples: (n,)
    Lx = jnp.dot(X, lambda_hat)              # (n, d)
    xLx = jnp.sum(X * Lx, axis=1)            # (n,)

    # w^T Λ̂ w for all prototypes: (p,)
    Lw = jnp.dot(W, lambda_hat)              # (p, d)
    wLw = jnp.sum(W * Lw, axis=1)            # (p,)

    # x^T Λ̂ w for all (n, p) pairs: (n, p)
    xLw = jnp.dot(Lx, W.T)                   # (n, p)

    # d_κ²(x, w) = exp(x^T Λ̂ x) + exp(w^T Λ̂ w) - 2·exp(x^T Λ̂ w)
    distances = (jnp.exp(xLx[:, None])
                 + jnp.exp(wLw[None, :])
                 - 2.0 * jnp.exp(xLw))

    return jnp.maximum(distances, 0.0)
