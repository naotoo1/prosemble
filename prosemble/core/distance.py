"""
JAX-based distance functions for Prosemble.

This module provides GPU-accelerated, vectorized distance computations
using JAX. All functions are JIT-compiled for maximum performance.

Mathematical Background
-----------------------
Distance metrics are fundamental to prototype-based learning algorithms.
This implementation focuses on:
1. Batch/matrix operations (no Python loops)
2. GPU compatibility
3. JIT compilation for speed
4. Numerical stability

Author: Prosemble Contributors
License: MIT
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import chex


# ============================================================================
# Core Distance Functions (Pairwise Matrices)
# ============================================================================


@jit
def euclidean_distance_matrix(X: chex.Array, Y: chex.Array) -> chex.Array:
    """
    Compute pairwise Euclidean distances between rows of X and Y.

    Uses the identity: ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
    This is more efficient than explicit broadcasting for large matrices.

    Mathematical Formula::

        D[i,j] = ||X[i] - Y[j]|| = sqrt(Σ_k (X[i,k] - Y[j,k])²)

    Args:
        X: Array of shape (n, d) - n samples with d features
        Y: Array of shape (m, d) - m samples with d features

    Returns:
        D: Array of shape (n, m) where ``D[i,j] = ||X[i] - Y[j]||``

    Complexity:
        Time: O(nmd) - single matrix multiplication
        Space: O(nm) - output matrix

    Example:
        >>> X = jnp.array([[0, 0], [1, 1], [2, 2]])
        >>> Y = jnp.array([[0, 0], [3, 3]])
        >>> D = euclidean_distance_matrix(X, Y)
        >>> D.shape
        (3, 2)
        >>> D[0, 0]  # Distance from X[0] to Y[0]
        0.0
        >>> D[2, 1]  # Distance from X[2] to Y[1]
        1.414...

    Notes:
        - Numerically stable: Uses maximum(D_sq, 0) to avoid sqrt of negatives
        - GPU-compatible: All operations are JAX primitives
        - JIT-compiled: First call compiles, subsequent calls are fast
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(X.shape[1], Y.shape[1])

    # Compute squared norms
    X_sq = jnp.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    Y_sq = jnp.sum(Y ** 2, axis=1, keepdims=True).T  # (1, m)

    # Compute dot product
    XY = X @ Y.T  # (n, m)

    # Apply distance formula
    D_sq = X_sq + Y_sq - 2 * XY

    # Ensure non-negative (numerical stability)
    D_sq = jnp.maximum(D_sq, 0.0)

    return jnp.sqrt(D_sq)


@jit
def squared_euclidean_distance_matrix(X: chex.Array, Y: chex.Array) -> chex.Array:
    """
    Compute pairwise squared Euclidean distances.

    More efficient than euclidean_distance_matrix(X, Y)**2 because it avoids
    the sqrt operation entirely.

    Mathematical Formula::

        D²[i,j] = ||X[i] - Y[j]||² = Σ_k (X[i,k] - Y[j,k])²

    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d)

    Returns:
        D²: Array of shape (n, m) where ``D²[i,j] = ||X[i] - Y[j]||²``

    Complexity:
        Time: O(nmd)
        Space: O(nm)

    Example:
        >>> X = jnp.array([[0, 0], [1, 1]])
        >>> Y = jnp.array([[0, 0], [2, 2]])
        >>> D_sq = squared_euclidean_distance_matrix(X, Y)
        >>> D_sq[1, 1]  # Squared distance from [1,1] to [2,2]
        2.0

    Notes:
        - Preferred over euclidean when squared distances are sufficient
        - Many algorithms (FCM, PCM) use squared distances directly
        - More numerically stable than squaring euclidean distances
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(X.shape[1], Y.shape[1])

    X_sq = jnp.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = jnp.sum(Y ** 2, axis=1, keepdims=True).T
    XY = X @ Y.T

    D_sq = X_sq + Y_sq - 2 * XY

    return jnp.maximum(D_sq, 0.0)


@jit
def manhattan_distance_matrix(X: chex.Array, Y: chex.Array) -> chex.Array:
    """
    Compute pairwise Manhattan (L1) distances.

    Mathematical Formula::

        D[i,j] = ||X[i] - Y[j]||₁ = Σ_k |X[i,k] - Y[j,k]|

    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d)

    Returns:
        D: Array of shape (n, m) where D[i,j] is Manhattan distance

    Complexity:
        Time: O(nmd)
        Space: O(nmd) - intermediate broadcasting

    Example:
        >>> X = jnp.array([[0, 0], [1, 1]])
        >>> Y = jnp.array([[0, 0], [2, 2]])
        >>> D = manhattan_distance_matrix(X, Y)
        >>> D[1, 1]  # Manhattan distance from [1,1] to [2,2]
        2.0

    Implementation:
        Uses broadcasting: X[:, None, :] - Y[None, :, :] creates (n, m, d)
        Then sums absolute differences along feature dimension.

    Notes:
        - Also known as "taxicab" or "city block" distance
        - More robust to outliers than Euclidean distance
        - Natural for sparse/binary features
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(X.shape[1], Y.shape[1])

    # Broadcasting: (n, 1, d) - (1, m, d) = (n, m, d)
    diff = X[:, None, :] - Y[None, :, :]

    # Sum absolute differences along feature dimension
    D = jnp.sum(jnp.abs(diff), axis=2)

    return D


def lpnorm_distance_matrix(
    X: chex.Array,
    Y: chex.Array,
    p: float | int
) -> chex.Array:
    """
    Compute pairwise L-p norm distances.

    Mathematical Formula::

        D[i,j] = ||X[i] - Y[j]||_p = (Σ_k |X[i,k] - Y[j,k]|^p)^(1/p)

    Special Cases:
        p = 1: Manhattan distance
        p = 2: Euclidean distance
        p = ∞: Chebyshev distance (max absolute difference)

    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d)
        p: Order of the norm (p >= 1)

    Returns:
        D: Array of shape (n, m) where D[i,j] is L-p distance

    Complexity:
        Time: O(nmd)
        Space: O(nmd)

    Example:
        >>> X = jnp.array([[0, 0], [1, 1]])
        >>> Y = jnp.array([[0, 0], [3, 4]])
        >>> D = lpnorm_distance_matrix(X, Y, p=2)  # Euclidean
        >>> D = lpnorm_distance_matrix(X, Y, p=1)  # Manhattan
        >>> D = lpnorm_distance_matrix(X, Y, p=jnp.inf)  # Chebyshev

    Notes:
        - For p=1, use manhattan_distance_matrix for better performance
        - For p=2, use euclidean_distance_matrix for better performance
        - For p=inf, computes ``max(|x - y|)``
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(X.shape[1], Y.shape[1])

    # Broadcasting
    diff = X[:, None, :] - Y[None, :, :]

    if p == jnp.inf:
        # Chebyshev distance: max absolute difference
        D = jnp.max(jnp.abs(diff), axis=2)
    else:
        # General L-p norm
        D = jnp.power(jnp.sum(jnp.power(jnp.abs(diff), p), axis=2), 1.0 / p)

    return D


@jit
def omega_distance_matrix(
    X: chex.Array,
    Y: chex.Array,
    omega: chex.Array
) -> chex.Array:
    """
    Compute distances in projected space using projection matrix Omega.

    Mathematical Formula::

        D[i,j] = ||X[i]Ω - Y[j]Ω||²

    where Ω is a projection matrix that transforms the feature space.

    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d)
        omega: Projection matrix of shape (d, k) where k is projection dimension

    Returns:
        D²: Array of shape (n, m) with squared distances in projected space

    Complexity:
        Time: O(ndk + mdk + nmk) = O((n+m)dk + nmk)
        Space: O(nk + mk + nm)

    Use Cases:
        - Dimensionality reduction for distance computation
        - Learning relevance of features (omega learned from data)
        - Mahalanobis-like distances (when omega = L where Σ = LL^T)

    Example:
        >>> X = jnp.array([[1, 2, 3], [4, 5, 6]])
        >>> Y = jnp.array([[0, 0, 0], [1, 1, 1]])
        >>> omega = jnp.array([[1, 0], [0, 1], [0, 0]])  # Project to first 2 dims
        >>> D = omega_distance_matrix(X, Y, omega)
        >>> D.shape
        (2, 2)

    Notes:
        - When omega is identity, reduces to squared Euclidean distance
        - When omega is learned, enables adaptive distance metrics
        - Used in GLVQ (Generalized Learning Vector Quantization)
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(omega, 2)
    chex.assert_equal(X.shape[1], omega.shape[0])
    chex.assert_equal(Y.shape[1], omega.shape[0])

    # Project data to new space
    X_proj = X @ omega  # (n, k)
    Y_proj = Y @ omega  # (m, k)

    # Compute distances in projected space
    D_sq = squared_euclidean_distance_matrix(X_proj, Y_proj)

    return D_sq


@jit
def lomega_distance_matrix(
    X: chex.Array,
    Y: chex.Array,
    omegas: chex.Array
) -> chex.Array:
    """
    Compute distances using multiple projection matrices (Local Omega).

    Mathematical Formula::

        D[i,j] = Σ_p ||X[i]Ω_p - Y[j]Ω_p||²

    where Ω_p are multiple projection matrices (one per prototype or cluster).

    Args:
        X: Array of shape (n, d) - data points
        Y: Array of shape (m, d) - prototypes/centroids
        omegas: Array of shape (m, d, k) - m projection matrices of size (d, k)
                Each Y[j] has its own projection matrix omegas[j]

    Returns:
        D²: Array of shape (n, m) with aggregated projected distances

    Complexity:
        Time: O(nmdk)
        Space: O(nmk)

    Use Cases:
        - Local relevance learning (each prototype has its own metric)
        - Adaptive distance metrics in GMLVQ
        - Cluster-specific feature weighting

    Example:
        >>> n, m, d, k = 10, 3, 5, 2
        >>> X = jax.random.normal(jax.random.PRNGKey(0), (n, d))
        >>> Y = jax.random.normal(jax.random.PRNGKey(1), (m, d))
        >>> omegas = jax.random.normal(jax.random.PRNGKey(2), (m, d, k))
        >>> D = lomega_distance_matrix(X, Y, omegas)
        >>> D.shape
        (10, 3)

    Implementation:
        Uses einsum for efficient tensor contraction:
        1. Project X through each omega: X @ omegas[j] for all j
        2. Extract diagonal for Y projections (each Y[j] uses omegas[j])
        3. Compute squared differences and sum

    Notes:
        - Generalizes omega_distance to local (per-prototype) metrics
        - More flexible but computationally expensive
        - Enables learning which features matter for each cluster
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(omegas, 3)
    chex.assert_equal(X.shape[1], omegas.shape[1])
    chex.assert_equal(Y.shape[1], omegas.shape[1])
    chex.assert_equal(Y.shape[0], omegas.shape[0])

    n, d = X.shape
    m, _, k = omegas.shape

    # Project X through all omegas: (n, m, d) @ (m, d, k) -> (n, m, k)
    # We need X[i] @ omegas[j] for all i, j
    X_expanded = X[:, None, :]  # (n, 1, d)
    X_proj = jnp.einsum('nid,mdk->nmk', X_expanded, omegas)  # (n, m, k)

    # Project Y through corresponding omegas: Y[j] @ omegas[j]
    # This is diagonal in the m dimension
    Y_proj = jnp.einsum('md,mdk->mk', Y, omegas)  # (m, k)

    # Compute squared differences: (n, m, k)
    # Broadcasting: (n, m, k) - (1, m, k)
    diff_sq = (X_proj - Y_proj[None, :, :]) ** 2

    # Sum over features and projection dimensions: (n, m, k) -> (n, m)
    D_sq = jnp.sum(diff_sq, axis=2)

    return D_sq


@jit
def tangent_distance_matrix(
    X: chex.Array,
    Y: chex.Array,
    omegas: chex.Array
) -> chex.Array:
    """
    Compute pairwise localized tangent distances.

    Each prototype j has an orthogonal subspace basis Omega_j of shape (d, s).
    The tangent distance projects out the subspace directions:

        d(x, w_j) = ||(I - Omega_j @ Omega_j^T)(x - w_j)||^2

    This is equivalent to:
        diff = x - w_j
        proj = Omega_j^T @ diff          (project onto subspace)
        recon = Omega_j @ proj            (reconstruct in ambient space)
        tangent_diff = diff - recon       (residual orthogonal to subspace)
        d = ||tangent_diff||^2

    Parameters
    ----------
    X : array of shape (n, d)
        Data points.
    Y : array of shape (m, d)
        Prototypes.
    omegas : array of shape (m, d, s)
        Orthogonal subspace bases per prototype, where s is the
        subspace dimension.

    Returns
    -------
    D : array of shape (n, m)
        Squared tangent distances.

    Notes
    -----
    Based on Saralajew, S., & Villmann, T. (2016). Adaptive tangent
    distances in generalized learning vector quantization.
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(omegas, 3)
    chex.assert_equal(X.shape[1], Y.shape[1])
    chex.assert_equal(Y.shape[0], omegas.shape[0])
    chex.assert_equal(Y.shape[1], omegas.shape[1])

    # diff: (n, m, d)
    diff = X[:, None, :] - Y[None, :, :]

    # Project onto each prototype's subspace: (n, m, d) @ (m, d, s) -> (n, m, s)
    proj = jnp.einsum('nmd,mds->nms', diff, omegas)

    # Reconstruct from subspace: (n, m, s) @ (m, s, d) -> (n, m, d)
    # omegas transposed: (m, d, s) -> (m, s, d)
    recon = jnp.einsum('nms,mds->nmd', proj, omegas)

    # Residual (orthogonal complement)
    tangent_diff = diff - recon

    # Squared norm
    return jnp.sum(tangent_diff ** 2, axis=2)


# ============================================================================
# Kernel Functions
# ============================================================================


@jit
def gaussian_kernel_matrix(
    X: chex.Array,
    Y: chex.Array,
    sigma: float
) -> chex.Array:
    """
    Compute Gaussian (RBF) kernel matrix.

    Mathematical Formula::

        K[i,j] = exp(-||X[i] - Y[j]||² / (2σ²))

    The Gaussian kernel maps data to infinite-dimensional Hilbert space,
    enabling non-linear clustering and classification.

    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d)
        sigma: Bandwidth parameter (σ > 0)

    Returns:
        K: Array of shape (n, m) where K[i,j] ∈ [0, 1]
           K[i,j] = 1 when X[i] = Y[j]
           K[i,j] → 0 as ||X[i] - Y[j]|| → ∞

    Complexity:
        Time: O(nmd)
        Space: O(nm)

    Properties:
        - K is positive semi-definite (valid kernel)
        - K is symmetric if X = Y
        - K[i,i] = 1 (self-similarity)

    Example:
        >>> X = jnp.array([[0, 0], [1, 1]])
        >>> Y = jnp.array([[0, 0], [2, 2]])
        >>> K = gaussian_kernel_matrix(X, Y, sigma=1.0)
        >>> K[0, 0]  # Self-similarity
        1.0
        >>> K[0, 1] < K[0, 0]  # Decreases with distance
        True

    Kernel Trick:
        For feature map φ: ℝ^d → ℋ (infinite-dimensional),
        K(x, y) = ⟨φ(x), φ(y)⟩ in Hilbert space ℋ

        Kernel distance:
            ||φ(x) - φ(y)||² = K(x,x) - 2K(x,y) + K(y,y)
                              = 2 - 2K(x,y)  [for normalized kernel]

    Use Cases:
        - Kernel Fuzzy C-Means (KFCM)
        - Kernel Possibilistic C-Means (KPCM)
        - Support Vector Machines (SVM)
        - Gaussian Processes

    Hyperparameter Tuning:
        - Small σ: Tight clusters, high sensitivity to noise
        - Large σ: Smooth clusters, may underfit
        - Rule of thumb: σ ≈ median(pairwise_distances) / √(2 * n_clusters)

    Notes:
        - sigma is bandwidth, NOT variance (variance = σ²)
        - For numerical stability, we use maximum() to ensure non-negative
        - JIT-compiled for GPU acceleration
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(X.shape[1], Y.shape[1])

    # Compute squared distances
    D_sq = squared_euclidean_distance_matrix(X, Y)

    # Apply Gaussian kernel
    K = jnp.exp(-D_sq / (2 * sigma ** 2))

    return K


@jit
def polynomial_kernel_matrix(
    X: chex.Array,
    Y: chex.Array,
    degree: int = 3,
    coef0: float = 1.0
) -> chex.Array:
    """
    Compute polynomial kernel matrix.

    Mathematical Formula::

        K[i,j] = (⟨X[i], Y[j]⟩ + c)^d

    where d is degree and c is coef0.

    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d)
        degree: Polynomial degree (d ≥ 1)
        coef0: Coefficient (c ≥ 0)

    Returns:
        K: Array of shape (n, m) with polynomial kernel values

    Example:
        >>> X = jnp.array([[1, 2], [3, 4]])
        >>> Y = jnp.array([[1, 0], [0, 1]])
        >>> K = polynomial_kernel_matrix(X, Y, degree=2, coef0=1.0)

    Notes:
        - degree=1, coef0=0: Linear kernel (⟨x, y⟩)
        - Higher degree: More complex decision boundaries
        - coef0: Influences importance of lower vs higher order terms
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(X.shape[1], Y.shape[1])

    # Compute dot products
    dot_products = X @ Y.T

    # Apply polynomial kernel
    K = jnp.power(dot_products + coef0, degree)

    return K


# ============================================================================
# Pairwise Distance Functions (for single pairs)
# ============================================================================


@jit
def euclidean_distance(x: chex.Array, y: chex.Array) -> chex.Array:
    """
    Euclidean distance between two vectors.

    Args:
        x: Array of shape (d,)
        y: Array of shape (d,)

    Returns:
        Scalar distance

    Example:
        >>> x = jnp.array([0, 0, 0])
        >>> y = jnp.array([1, 1, 1])
        >>> d = euclidean_distance(x, y)
        >>> d
        Array(1.732..., dtype=float32)
    """
    chex.assert_equal_shape([x, y])
    return jnp.sqrt(jnp.sum((x - y) ** 2))


@jit
def squared_euclidean_distance(x: chex.Array, y: chex.Array) -> chex.Array:
    """
    Squared Euclidean distance between two vectors.

    Args:
        x: Array of shape (d,)
        y: Array of shape (d,)

    Returns:
        Scalar squared distance
    """
    chex.assert_equal_shape([x, y])
    return jnp.sum((x - y) ** 2)


@jit
def manhattan_distance(x: chex.Array, y: chex.Array) -> chex.Array:
    """
    Manhattan (L1) distance between two vectors.

    Args:
        x: Array of shape (d,)
        y: Array of shape (d,)

    Returns:
        Scalar distance
    """
    chex.assert_equal_shape([x, y])
    return jnp.sum(jnp.abs(x - y))


def lpnorm_distance(x: chex.Array, y: chex.Array, p: float = 2) -> chex.Array:
    """
    Lp-norm distance between two vectors.

    Args:
        x: Array of shape (d,)
        y: Array of shape (d,)
        p: Order of the norm (supports inf)

    Returns:
        Scalar distance
    """
    chex.assert_equal_shape([x, y])
    return jnp.linalg.norm(x - y, ord=p)


@jit
def omega_distance(x: chex.Array, y: chex.Array, omega: chex.Array) -> chex.Array:
    """
    Omega (projection-based) distance between two vectors.

    Computes ||diff @ omega||² where diff = x - y.

    Args:
        x: Array of shape (d,)
        y: Array of shape (d,)
        omega: Projection matrix of shape (d, k)

    Returns:
        Scalar squared distance in projected space
    """
    chex.assert_equal_shape([x, y])
    diff = x - y
    projected = diff @ omega
    return jnp.sum(projected ** 2)


def lomega_distance(X: chex.Array, Y: chex.Array, omegas: chex.Array) -> chex.Array:
    """
    Local omega distance with per-prototype projection matrices.

    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d) — prototypes
        omegas: Array of shape (m, d, k) — one projection matrix per prototype

    Returns:
        Distance matrix of shape (n, m)
    """
    def compute_single(x, y, omega):
        diff = x - y
        projected = diff @ omega
        return jnp.sum(projected ** 2)

    def compute_row(x):
        return jax.vmap(compute_single, in_axes=(None, 0, 0))(x, Y, omegas)

    return jax.vmap(compute_row)(X)


# ============================================================================
# Utility Functions
# ============================================================================


def estimate_sigma(X: chex.Array, percentile: float = 50.0) -> float:
    """
    Estimate sigma for Gaussian kernel using pairwise distances.

    Strategy: Use median (or other percentile) of pairwise distances.

    Args:
        X: Data array of shape (n, d)
        percentile: Percentile of distances to use (0-100)

    Returns:
        sigma: Estimated bandwidth parameter

    Example:
        >>> X = jax.random.normal(jax.random.PRNGKey(0), (100, 10))
        >>> sigma = estimate_sigma(X, percentile=50)

    Notes:
        - Heuristic: sigma = median_distance / √(2 * n_clusters)
        - For large datasets, use subsample to avoid O(n²) computation
    """
    # For large datasets, subsample
    n = X.shape[0]
    if n > 1000:
        key = jax.random.PRNGKey(0)
        indices = jax.random.choice(key, n, shape=(1000,), replace=False)
        X_sub = X[indices]
    else:
        X_sub = X

    # Compute pairwise distances
    D = euclidean_distance_matrix(X_sub, X_sub)

    # Get upper triangle (exclude diagonal and duplicates)
    mask = jnp.triu(jnp.ones_like(D, dtype=bool), k=1)
    distances = D[mask]

    # Compute percentile
    sigma = jnp.percentile(distances, percentile)

    return float(sigma)


@jit
def safe_divide(numerator: chex.Array, denominator: chex.Array, epsilon: float = 1e-10) -> chex.Array:
    """
    Safe division avoiding division by zero.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        epsilon: Small value to add to denominator

    Returns:
        numerator / (denominator + epsilon)

    Example:
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = jnp.array([2.0, 0.0, 1.0])
        >>> safe_divide(x, y)
        Array([0.5, 2e+09, 3.0], dtype=float32)  # Avoids inf
    """
    return numerator / (denominator + epsilon)


# ============================================================================
# Module Information
# ============================================================================

# Aliases for convenience
batch_squared_euclidean = squared_euclidean_distance_matrix
batch_euclidean = euclidean_distance_matrix

__all__ = [
    # Matrix distance functions
    'euclidean_distance_matrix',
    'squared_euclidean_distance_matrix',
    'manhattan_distance_matrix',
    'lpnorm_distance_matrix',
    'omega_distance_matrix',
    'lomega_distance_matrix',
    'tangent_distance_matrix',
    # Kernel functions
    'gaussian_kernel_matrix',
    'polynomial_kernel_matrix',
    # Pairwise functions
    'euclidean_distance',
    'squared_euclidean_distance',
    'manhattan_distance',
    'lpnorm_distance',
    'omega_distance',
    'lomega_distance',
    # Utilities
    'estimate_sigma',
    'safe_divide',
    # Aliases
    'batch_squared_euclidean',
    'batch_euclidean',
]
