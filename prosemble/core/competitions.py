"""
Competition mechanisms for prototype-based classification.

These functions determine class predictions from distance matrices
and prototype labels using different strategies.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial


@jit
def wtac(distances, prototype_labels):
    """Winner-Takes-All Competition.

    Assigns each sample the label of the closest prototype.

    Parameters
    ----------
    distances : array of shape (n_samples, n_prototypes)
        Distance matrix.
    prototype_labels : array of shape (n_prototypes,)
        Class label for each prototype.

    Returns
    -------
    array of shape (n_samples,)
        Predicted labels.
    """
    winning_indices = jnp.argmin(distances, axis=1)
    return prototype_labels[winning_indices]


@partial(jit, static_argnums=(2, 3))
def knnc(distances, prototype_labels, k=1, n_classes=None):
    """K-Nearest Neighbors Competition.

    Assigns each sample the majority label among k closest prototypes.

    Parameters
    ----------
    distances : array of shape (n_samples, n_prototypes)
        Distance matrix.
    prototype_labels : array of shape (n_prototypes,)
        Class label for each prototype.
    k : int
        Number of neighbors.
    n_classes : int or None
        Number of classes. If None, inferred from prototype_labels.

    Returns
    -------
    array of shape (n_samples,)
        Predicted labels.
    """
    sorted_indices = jnp.argsort(distances, axis=1)[:, :k]  # (n, k)
    k_labels = prototype_labels[sorted_indices]  # (n, k)

    # Majority vote via one-hot counting
    if n_classes is None:
        n_classes = int(jnp.max(prototype_labels)) + 1

    def _vote(labels_row):
        one_hot = jnp.eye(n_classes, dtype=jnp.int32)[labels_row]
        counts = jnp.sum(one_hot, axis=0)
        return jnp.argmax(counts)

    import jax
    return jax.vmap(_vote)(k_labels)


@jit
def cbcc(detections, reasonings):
    """Classification-By-Components Competition.

    Computes class probability distributions using component detections
    and reasoning matrices.

    Parameters
    ----------
    detections : array of shape (n_samples, n_components)
        Similarity/detection scores for each component.
    reasonings : array of shape (n_components, n_classes, 2)
        Reasoning matrices. Last dim: [positive, negative_raw].

    Returns
    -------
    array of shape (n_samples, n_classes)
        Class probability distributions.
    """
    # Extract positive and negative reasoning
    # A = raw positive, B = raw negative factor
    A = jnp.clip(reasonings[:, :, 0], 0.0, 1.0)  # (n_comp, n_classes)
    B = jnp.clip(reasonings[:, :, 1], 0.0, 1.0)  # (n_comp, n_classes)

    pk = A  # positive reasoning
    nk = (1.0 - A) * B  # negative reasoning

    # numerator: detections @ (pk - nk)^T + sum(nk, axis=0)
    # detections: (n, n_comp), (pk - nk): (n_comp, n_classes)
    numerator = detections @ (pk - nk) + jnp.sum(nk, axis=0)

    # denominator: sum(pk + nk, axis=0) + epsilon
    denominator = jnp.sum(pk + nk, axis=0) + 1e-8

    probs = numerator / denominator
    return probs
