"""
Data loading and batching utilities for prosemble.

Provides composable primitives for mini-batch iteration, suitable for
custom training loops.  The built-in ``fit()`` methods already handle
batching internally — these utilities are for advanced users who need
explicit control over data iteration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def shuffle_arrays(key: jax.Array, *arrays: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """Shuffle multiple arrays with the same random permutation.

    Parameters
    ----------
    key : JAX PRNG key
        Random key for generating the permutation.
    *arrays : jnp.ndarray
        Arrays to shuffle.  All must have the same length along axis 0.

    Returns
    -------
    tuple of jnp.ndarray
        Shuffled arrays in the same order as the inputs.

    Examples
    --------
    >>> key = jax.random.PRNGKey(0)
    >>> X = jnp.arange(12).reshape(4, 3)
    >>> y = jnp.array([0, 1, 0, 1])
    >>> X_s, y_s = shuffle_arrays(key, X, y)
    """
    if len(arrays) == 0:
        return ()
    n = arrays[0].shape[0]
    perm = jax.random.permutation(key, n)
    return tuple(a[perm] for a in arrays)


def padded_batches(
    X: jnp.ndarray,
    y: jnp.ndarray | None = None,
    batch_size: int = 32,
    key: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Split data into static-shaped batches, padding the last batch.

    Returns arrays of shape ``(n_batches, batch_size, ...)`` suitable for
    ``jax.lax.scan``.  If the data length is not divisible by
    ``batch_size``, the last batch is padded by repeating initial samples.

    Parameters
    ----------
    X : jnp.ndarray of shape (n_samples, ...)
        Feature array.
    y : jnp.ndarray of shape (n_samples,), optional
        Label array.
    batch_size : int
        Number of samples per batch.
    key : JAX PRNG key, optional
        If provided, data is shuffled before batching.

    Returns
    -------
    X_batches : jnp.ndarray of shape (n_batches, batch_size, ...)
    y_batches : jnp.ndarray of shape (n_batches, batch_size) or None

    Examples
    --------
    >>> X = jnp.ones((10, 3))
    >>> y = jnp.arange(10)
    >>> X_b, y_b = padded_batches(X, y, batch_size=4)
    >>> X_b.shape  # (3, 4, 3) — 10 samples padded to 12
    (3, 4, 3)
    """
    n_samples = X.shape[0]

    if key is not None:
        if y is not None:
            X, y = shuffle_arrays(key, X, y)
        else:
            X, = shuffle_arrays(key, X)

    n_batches = (n_samples + batch_size - 1) // batch_size
    padded_size = n_batches * batch_size

    if padded_size > n_samples:
        pad_n = padded_size - n_samples
        X = jnp.concatenate([X, X[:pad_n]], axis=0)
        if y is not None:
            y = jnp.concatenate([y, y[:pad_n]], axis=0)

    X_batches = X.reshape(n_batches, batch_size, *X.shape[1:])
    y_batches = y.reshape(n_batches, batch_size) if y is not None else None
    return X_batches, y_batches


def batched_iterator(
    X: jnp.ndarray,
    y: jnp.ndarray | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    key: jax.Array | None = None,
    drop_last: bool = False,
):
    """Yield mini-batches from data arrays.

    For use in custom Python training loops.

    Parameters
    ----------
    X : jnp.ndarray of shape (n_samples, ...)
        Feature array.
    y : jnp.ndarray of shape (n_samples,), optional
        Label array.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        If True and *key* is provided, shuffle before iterating.
    key : JAX PRNG key, optional
        Required if ``shuffle=True``.
    drop_last : bool
        If True, drop the last batch when it is smaller than
        ``batch_size``.  If False (default), the last batch may be
        smaller.

    Yields
    ------
    X_batch : jnp.ndarray of shape (batch_size, ...) or smaller
    y_batch : jnp.ndarray of shape (batch_size,) or None

    Examples
    --------
    >>> key = jax.random.PRNGKey(0)
    >>> X = jnp.ones((10, 3))
    >>> for X_b, y_b in batched_iterator(X, batch_size=4, key=key):
    ...     print(X_b.shape)
    (4, 3)
    (4, 3)
    (2, 3)
    """
    n_samples = X.shape[0]

    if shuffle and key is not None:
        if y is not None:
            X, y = shuffle_arrays(key, X, y)
        else:
            X, = shuffle_arrays(key, X)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        if drop_last and (end - start) < batch_size:
            break
        X_batch = X[start:end]
        y_batch = y[start:end] if y is not None else None
        yield X_batch, y_batch
