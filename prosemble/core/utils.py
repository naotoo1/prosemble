"""
JAX utility functions for data preprocessing and manipulation.

Replaces common sklearn/numpy operations with JAX-native implementations.
"""

import jax
import jax.numpy as jnp
import chex


def train_test_split_jax(
    X: chex.Array,
    y: chex.Array | None = None,
    test_size: float = 0.2,
    random_seed: int = 42
) -> tuple[chex.Array, ...]:
    """
    JAX-native train/test split.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,), optional
        Labels
    test_size : float, default=0.2
        Proportion of dataset for test set (0.0 to 1.0)
    random_seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X_train, X_test[, y_train, y_test]
        Split arrays. If y is provided, returns 4 arrays, else 2.

    Examples
    --------
    >>> X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = jnp.array([0, 1, 0, 1])
    >>> X_train, X_test, y_train, y_test = train_test_split_jax(X, y, test_size=0.5)
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # Shuffle indices
    key = jax.random.PRNGKey(random_seed)
    indices = jax.random.permutation(key, n_samples)

    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]

    if y is not None:
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test


def standardize(X: chex.Array, mean: chex.Array | None = None,
                std: chex.Array | None = None) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Standardize features (zero mean, unit variance).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to standardize
    mean : array-like of shape (n_features,), optional
        Pre-computed mean (for test data)
    std : array-like of shape (n_features,), optional
        Pre-computed std (for test data)

    Returns
    -------
    X_scaled : array-like
        Standardized data
    mean : array-like
        Mean used for scaling
    std : array-like
        Std used for scaling

    Examples
    --------
    >>> X_train = jnp.array([[1, 2], [3, 4], [5, 6]])
    >>> X_scaled, mean, std = standardize(X_train)
    >>> # For test data
    >>> X_test_scaled, _, _ = standardize(X_test, mean=mean, std=std)
    """
    if mean is None:
        mean = jnp.mean(X, axis=0)
    if std is None:
        std = jnp.std(X, axis=0)

    # Avoid division by zero
    std = jnp.where(std == 0, 1.0, std)

    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def min_max_scale(X: chex.Array, min_val: chex.Array | None = None,
                  max_val: chex.Array | None = None) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Scale features to [0, 1] range.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to scale
    min_val : array-like of shape (n_features,), optional
        Pre-computed min (for test data)
    max_val : array-like of shape (n_features,), optional
        Pre-computed max (for test data)

    Returns
    -------
    X_scaled : array-like
        Scaled data
    min_val : array-like
        Min values used
    max_val : array-like
        Max values used
    """
    if min_val is None:
        min_val = jnp.min(X, axis=0)
    if max_val is None:
        max_val = jnp.max(X, axis=0)

    # Avoid division by zero
    range_val = max_val - min_val
    range_val = jnp.where(range_val == 0, 1.0, range_val)

    X_scaled = (X - min_val) / range_val
    return X_scaled, min_val, max_val


def pca_jax(X: chex.Array, n_components: int = 2) -> tuple[chex.Array, chex.Array]:
    """
    Principal Component Analysis using JAX.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix
    n_components : int, default=2
        Number of principal components

    Returns
    -------
    X_transformed : array-like of shape (n_samples, n_components)
        Transformed data
    components : array-like of shape (n_components, n_features)
        Principal components

    Examples
    --------
    >>> X = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> X_pca, components = pca_jax(X, n_components=2)
    """
    # Center data
    X_centered = X - jnp.mean(X, axis=0)

    # Compute covariance matrix
    cov_matrix = jnp.cov(X_centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)

    # Sort by eigenvalues (descending)
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    components = eigenvectors[:, :n_components].T

    # Transform data
    X_transformed = X_centered @ components.T

    return X_transformed, components


def accuracy_score_jax(y_true: chex.Array, y_pred: chex.Array) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels
    y_pred : array-like of shape (n_samples,)
        Predicted labels

    Returns
    -------
    accuracy : float
        Accuracy score

    Examples
    --------
    >>> y_true = jnp.array([0, 1, 1, 0])
    >>> y_pred = jnp.array([0, 1, 0, 0])
    >>> accuracy_score_jax(y_true, y_pred)
    0.75
    """
    return float(jnp.mean(y_true == y_pred))


def confusion_matrix_jax(y_true: chex.Array, y_pred: chex.Array, n_classes: int) -> chex.Array:
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels
    y_pred : array-like of shape (n_samples,)
        Predicted labels
    n_classes : int
        Number of classes

    Returns
    -------
    conf_matrix : array-like of shape (n_classes, n_classes)
        Confusion matrix

    Examples
    --------
    >>> y_true = jnp.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = jnp.array([0, 2, 2, 0, 0, 1])
    >>> confusion_matrix_jax(y_true, y_pred, n_classes=3)
    """
    conf_matrix = jnp.zeros((n_classes, n_classes), dtype=jnp.int32)

    for i in range(n_classes):
        for j in range(n_classes):
            conf_matrix = conf_matrix.at[i, j].set(
                jnp.sum((y_true == i) & (y_pred == j))
            )

    return conf_matrix


def shuffle_jax(X: chex.Array, y: chex.Array | None = None,
                random_seed: int = 42) -> tuple[chex.Array, ...]:
    """
    Shuffle arrays in unison.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,), optional
        Labels
    random_seed : int, default=42
        Random seed

    Returns
    -------
    X_shuffled[, y_shuffled]
        Shuffled arrays

    Examples
    --------
    >>> X = jnp.array([[1, 2], [3, 4], [5, 6]])
    >>> y = jnp.array([0, 1, 0])
    >>> X_shuf, y_shuf = shuffle_jax(X, y, random_seed=42)
    """
    n_samples = len(X)
    key = jax.random.PRNGKey(random_seed)
    indices = jax.random.permutation(key, n_samples)

    X_shuffled = X[indices]

    if y is not None:
        y_shuffled = y[indices]
        return X_shuffled, y_shuffled
    else:
        return (X_shuffled,)


def k_fold_split_jax(n_samples: int, n_folds: int = 5, random_seed: int = 42):
    """
    Generate K-fold cross-validation indices.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_folds : int, default=5
        Number of folds
    random_seed : int, default=42
        Random seed

    Yields
    ------
    train_indices, test_indices
        Indices for each fold

    Examples
    --------
    >>> for train_idx, test_idx in k_fold_split_jax(100, n_folds=5):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    """
    key = jax.random.PRNGKey(random_seed)
    indices = jax.random.permutation(key, n_samples)

    fold_size = n_samples // n_folds

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else n_samples

        test_indices = indices[test_start:test_end]
        train_indices = jnp.concatenate([
            indices[:test_start],
            indices[test_end:]
        ])

        yield train_indices, test_indices


def orthogonalize(matrix):
    """Orthogonalize a matrix via polar decomposition (SVD).

    Given a matrix A of shape (d, s), computes the closest orthogonal
    matrix Q such that Q^T Q = I, using the polar factor:

        U, S, V^T = SVD(A)
        Q = U @ V^T

    Supports batched input via jax.vmap.

    Parameters
    ----------
    matrix : array of shape (d, s) or (n, d, s)
        Matrix or batch of matrices to orthogonalize.
        For batched input, use ``jax.vmap(orthogonalize)``.

    Returns
    -------
    Q : array of same shape as input
        Orthogonal matrix (columns are orthonormal).
    """
    U, _, Vt = jnp.linalg.svd(matrix, full_matrices=False)
    return U @ Vt


def class_priors(labels, n_classes=None):
    """Compute class prior probabilities from labels.

    P(class=k) = n_k / n

    Parameters
    ----------
    labels : array of shape (n_samples,)
        Integer class labels.
    n_classes : int, optional
        Number of classes. Inferred from labels if not provided.

    Returns
    -------
    priors : array of shape (n_classes,)
        Prior probability for each class, sums to 1.
    """
    labels = jnp.asarray(labels, dtype=jnp.int32)
    if n_classes is None:
        n_classes = int(jnp.max(labels)) + 1
    counts = jnp.zeros(n_classes)
    for c in range(n_classes):
        counts = counts.at[c].set(jnp.sum(labels == c))
    return counts / jnp.sum(counts)


def prototype_priors(prototype_labels, n_classes=None):
    """Compute class priors from prototype label distribution.

    Used in probabilistic LVQ models where the prior over prototypes
    is uniform (1/n_prototypes) and the class prior is the fraction
    of prototypes assigned to each class.

    ``P(class=k) = |{j : label_j = k}| / n_prototypes``

    Parameters
    ----------
    prototype_labels : array of shape (n_prototypes,)
        Class label for each prototype.
    n_classes : int, optional
        Number of classes. Inferred if not provided.

    Returns
    -------
    priors : array of shape (n_classes,)
        Class prior probabilities, sums to 1.
    """
    return class_priors(prototype_labels, n_classes)


def uniform_prototype_prior(n_prototypes):
    """Uniform prior over prototypes: P(prototype_j) = 1/n.

    This is the standard prior used in probabilistic LVQ (SLVQ/RSLVQ).

    Parameters
    ----------
    n_prototypes : int
        Number of prototypes.

    Returns
    -------
    prior : array of shape (n_prototypes,)
        Uniform probability vector, sums to 1.
    """
    return jnp.ones(n_prototypes) / n_prototypes
