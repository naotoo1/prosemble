"""
Similarity functions for prototype-based learning.

Similarities are the dual of distances: higher values indicate
closer/more similar points.
"""

import jax.numpy as jnp
from jax import jit


@jit
def gaussian_similarity(distances_sq, variance=1.0):
    """Convert squared distances to Gaussian similarities.

    s(d) = exp(-d^2 / (2 * variance))

    Parameters
    ----------
    distances_sq : array
        Squared distances.
    variance : float
        Variance (sigma^2) of the Gaussian.

    Returns
    -------
    array
        Similarity values in (0, 1].
    """
    return jnp.exp(-distances_sq / (2.0 * variance))


@jit
def cosine_similarity_matrix(X, Y):
    """Pairwise cosine similarity between rows of X and Y.

    cos(x, y) = (x . y) / (||x|| * ||y||)

    Parameters
    ----------
    X : array of shape (n, d)
    Y : array of shape (m, d)

    Returns
    -------
    array of shape (n, m)
        Cosine similarities in [-1, 1].
    """
    # Compute norms
    norm_X = jnp.linalg.norm(X, axis=1, keepdims=True)  # (n, 1)
    norm_Y = jnp.linalg.norm(Y, axis=1, keepdims=True)  # (m, 1)

    # Avoid division by zero
    eps = jnp.finfo(X.dtype).eps
    norm_X = jnp.maximum(norm_X, eps)
    norm_Y = jnp.maximum(norm_Y, eps)

    # Dot product matrix / outer product of norms
    dot_product = X @ Y.T  # (n, m)
    norm_product = norm_X @ norm_Y.T  # (n, m)

    return dot_product / norm_product


@jit
def euclidean_similarity(X, Y, variance=1.0):
    """Pairwise Euclidean similarity (Gaussian of Euclidean distance).

    Parameters
    ----------
    X : array of shape (n, d)
    Y : array of shape (m, d)
    variance : float
        Variance of the Gaussian kernel.

    Returns
    -------
    array of shape (n, m)
        Similarity values in (0, 1].
    """
    diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
    dist_sq = jnp.sum(diff ** 2, axis=2)  # (n, m)
    return jnp.exp(-dist_sq / (2.0 * variance))


@jit
def rank_scaled_gaussian(distances, lambd=1.0):
    """Rank-scaled Gaussian similarity.

    Combines distance magnitude with rank ordering: closer prototypes
    (lower rank) receive a stronger signal, while farther ones are
    exponentially suppressed.

    .. math::

        s(d, r) = \exp(-\exp(-r / \lambda) \cdot d)

    where *r* is the rank of each distance (0 = closest).

    Parameters
    ----------
    distances : array of shape (n, m)
        Distance matrix (non-negative).
    lambd : float
        Rank decay parameter. Larger values give more uniform weighting
        across ranks; smaller values concentrate on nearest neighbours.

    Returns
    -------
    array of shape (n, m)
        Rank-scaled similarity values in (0, 1].

    Notes
    -----
    Used in Probabilistic LVQ (PLVQ) as a conditional
    distribution P(x|prototype).
    """
    order = jnp.argsort(distances, axis=1)
    ranks = jnp.argsort(order, axis=1).astype(distances.dtype)
    return jnp.exp(-jnp.exp(-ranks / lambd) * distances)
