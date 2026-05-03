"""
Stratified pooling operations for prototype-based learning.

These functions aggregate per-prototype distances into per-class distances,
grouping by prototype label. Essential for GLVQ and CELVQ.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnums=(2,))
def stratified_min_pooling(distances, prototype_labels, n_classes):
    """Per-class minimum distance pooling.

    For each sample and each class, returns the minimum distance
    to any prototype of that class.

    Parameters
    ----------
    distances : array of shape (n_samples, n_prototypes)
        Distance matrix.
    prototype_labels : array of shape (n_prototypes,)
        Class label for each prototype.
    n_classes : int
        Number of classes.

    Returns
    -------
    array of shape (n_samples, n_classes)
        Minimum distance to each class for each sample.
    """
    # class_mask[c, j] = True if prototype j belongs to class c
    class_mask = (prototype_labels[None, :] == jnp.arange(n_classes)[:, None])
    INF = jnp.finfo(distances.dtype).max
    # masked[i, c, j] = distances[i, j] if proto j is class c, else INF
    masked = jnp.where(class_mask[None, :, :], distances[:, None, :], INF)
    return jnp.min(masked, axis=2)


@partial(jit, static_argnums=(2,))
def stratified_sum_pooling(distances, prototype_labels, n_classes):
    """Per-class sum distance pooling.

    Parameters
    ----------
    distances : array of shape (n_samples, n_prototypes)
    prototype_labels : array of shape (n_prototypes,)
    n_classes : int

    Returns
    -------
    array of shape (n_samples, n_classes)
        Sum of distances to each class for each sample.
    """
    class_mask = (prototype_labels[None, :] == jnp.arange(n_classes)[:, None])
    masked = jnp.where(class_mask[None, :, :], distances[:, None, :], 0.0)
    return jnp.sum(masked, axis=2)


@partial(jit, static_argnums=(2,))
def stratified_max_pooling(distances, prototype_labels, n_classes):
    """Per-class maximum distance pooling.

    Parameters
    ----------
    distances : array of shape (n_samples, n_prototypes)
    prototype_labels : array of shape (n_prototypes,)
    n_classes : int

    Returns
    -------
    array of shape (n_samples, n_classes)
        Maximum distance to each class for each sample.
    """
    class_mask = (prototype_labels[None, :] == jnp.arange(n_classes)[:, None])
    NEG_INF = -jnp.finfo(distances.dtype).max
    masked = jnp.where(class_mask[None, :, :], distances[:, None, :], NEG_INF)
    return jnp.max(masked, axis=2)


@partial(jit, static_argnums=(2,))
def stratified_prod_pooling(distances, prototype_labels, n_classes):
    """Per-class product distance pooling.

    Uses log-sum-exp for numerical stability.

    Parameters
    ----------
    distances : array of shape (n_samples, n_prototypes)
    prototype_labels : array of shape (n_prototypes,)
    n_classes : int

    Returns
    -------
    array of shape (n_samples, n_classes)
        Product of distances to each class for each sample.
    """
    class_mask = (prototype_labels[None, :] == jnp.arange(n_classes)[:, None])
    # For product, use log-sum approach: prod = exp(sum(log(x)))
    # Replace non-class entries with 1.0 (log(1) = 0, neutral for sum)
    masked = jnp.where(class_mask[None, :, :], distances[:, None, :], 1.0)
    log_masked = jnp.log(jnp.maximum(masked, 1e-30))
    return jnp.exp(jnp.sum(
        jnp.where(class_mask[None, :, :], log_masked, 0.0), axis=2
    ))
