"""Regularization techniques for prototype-based models.

Provides reusable loss-level regularization terms that can be composed
with any model's ``_compute_loss`` method:

- ``prototype_diversity_loss``: DPP-inspired repulsion between same-class
  prototypes.
- ``sparse_relevance_proximal``: Elastic net proximal step for relevance
  vectors (soft-thresholding).

References
----------
.. [1] Kulesza, A. & Taskar, B. (2012). Determinantal Point Processes for
       Machine Learning. Foundations and Trends in Machine Learning.
.. [2] Tibshirani, R. (1996). Regression shrinkage and selection via the
       LASSO. JRSS-B.
"""

import jax
import jax.numpy as jnp


def prototype_diversity_loss(prototypes, proto_labels, sigma_div=1.0):
    """Compute DPP-inspired diversity regularization for same-class prototypes.

    Encourages multiple prototypes per class to spread out rather than
    collapsing to the same point. Uses the log-determinant of the
    RBF kernel matrix between same-class prototypes.

    .. math::

        L_{\\text{diversity}} = -\\sum_c \\log\\det(K_c + \\epsilon I)

    where :math:`K_c[i,j] = \\exp(-\\|w_i - w_j\\|^2 / (2\\sigma^2))`.

    The determinant is maximized when prototypes are spread out (DPP
    theory). When two prototypes collapse, det -> 0, so -log(det) -> inf.

    Parameters
    ----------
    prototypes : array of shape (n_prototypes, n_features)
        Prototype positions.
    proto_labels : array of shape (n_prototypes,)
        Class label for each prototype.
    sigma_div : float
        Bandwidth for the RBF kernel. Controls the scale at which
        diversity is measured. Default: 1.0.

    Returns
    -------
    loss : scalar
        Diversity penalty (lower is more diverse).

    Notes
    -----
    Only active when n_prototypes_per_class > 1. For single-prototype
    classes, the contribution is zero (log(det(1x1 identity)) = 0).
    """
    n_protos = prototypes.shape[0]
    classes = jnp.unique(proto_labels, size=int(jnp.max(proto_labels)) + 1)
    n_classes = classes.shape[0]

    # Compute pairwise squared distances between all prototypes
    diff = prototypes[:, None, :] - prototypes[None, :, :]  # (P, P, D)
    sq_dists = jnp.sum(diff ** 2, axis=2)  # (P, P)

    # RBF kernel matrix
    K = jnp.exp(-sq_dists / (2.0 * sigma_div ** 2))

    # Add small diagonal for numerical stability
    K = K + 1e-6 * jnp.eye(n_protos)

    # Sum log-determinants per class
    total_loss = 0.0
    for c_idx in range(n_classes):
        c = classes[c_idx]
        mask = (proto_labels == c)
        n_c = jnp.sum(mask).astype(jnp.int32)

        # Extract sub-kernel for this class
        # Use masking approach for JIT compatibility
        indices = jnp.where(mask, size=n_protos, fill_value=0)[0]
        K_c = K[indices[:, None], indices[None, :]]

        # Only compute logdet if more than 1 prototype for this class
        # For single prototype, K_c = [[1+eps]], logdet ≈ 0
        sign, logdet = jnp.linalg.slogdet(K_c)
        # Negative because we want to maximize det (minimize -logdet)
        total_loss = total_loss + jnp.where(n_c > 1, -logdet, 0.0)

    return total_loss


def prototype_diversity_loss_vectorized(prototypes, proto_labels, sigma_div=1.0,
                                        n_classes=None, max_protos_per_class=None):
    """JIT-friendly vectorized diversity loss without Python loops.

    For use within ``lax.scan`` training (``use_scan=True``). Requires
    static shape information.

    Parameters
    ----------
    prototypes : array of shape (n_prototypes, n_features)
    proto_labels : array of shape (n_prototypes,)
    sigma_div : float
        RBF bandwidth. Default: 1.0.
    n_classes : int
        Number of classes (must be known at compile time).
    max_protos_per_class : int
        Maximum prototypes in any class (must be known at compile time).

    Returns
    -------
    loss : scalar
    """
    n_protos = prototypes.shape[0]

    # Pairwise squared distances
    diff = prototypes[:, None, :] - prototypes[None, :, :]
    sq_dists = jnp.sum(diff ** 2, axis=2)
    K_full = jnp.exp(-sq_dists / (2.0 * sigma_div ** 2))
    K_full = K_full + 1e-6 * jnp.eye(n_protos)

    total_loss = jnp.array(0.0)

    def class_logdet(c, carry):
        mask = (proto_labels == c)
        # Gather indices for this class (padded)
        indices = jnp.where(mask, size=max_protos_per_class, fill_value=0)[0]
        K_c = K_full[indices[:, None], indices[None, :]]
        # Mask out padding rows/cols with identity
        n_c = jnp.sum(mask)
        valid = jnp.arange(max_protos_per_class) < n_c
        valid_mask = valid[:, None] & valid[None, :]
        K_c = jnp.where(valid_mask, K_c, jnp.eye(max_protos_per_class))
        _, logdet = jnp.linalg.slogdet(K_c)
        return carry + jnp.where(n_c > 1, -logdet, 0.0)

    from jax import lax
    total_loss = lax.fori_loop(0, n_classes, class_logdet, total_loss)
    return total_loss


def sparse_relevance_proximal(relevances, l1_weight, lr=1.0):
    """Apply proximal operator for L1 regularization (soft-thresholding).

    Enforces true sparsity in relevance vectors by applying the
    proximal mapping of the L1 norm after the gradient step:

    .. math::

        \\text{prox}_{\\alpha\\|\\cdot\\|_1}(x)_j =
            \\text{sign}(x_j) \\cdot \\max(|x_j| - \\alpha, 0)

    This gives genuinely sparse feature selection with LASSO consistency
    guarantees.

    Parameters
    ----------
    relevances : array of shape (n_features,) or (n_prototypes, n_features)
        Relevance logits (before softmax normalization).
    l1_weight : float
        L1 penalty strength. Larger values give sparser solutions.
    lr : float
        Learning rate (used to scale the threshold: threshold = l1_weight * lr).
        Default: 1.0.

    Returns
    -------
    sparse_relevances : array, same shape as input
        Thresholded relevances with exact zeros.
    """
    threshold = l1_weight * lr
    return jnp.sign(relevances) * jnp.maximum(jnp.abs(relevances) - threshold, 0.0)


def elastic_net_proximal(relevances, l1_weight, l2_weight, lr=1.0):
    """Apply elastic net proximal operator (L1 + L2 regularization).

    Combines soft-thresholding (L1 sparsity) with L2 shrinkage:

    .. math::

        \\text{prox}(x)_j = \\frac{\\text{sign}(x_j) \\max(|x_j| - \\alpha, 0)}
                                   {1 + \\beta}

    where alpha = l1_weight * lr, beta = l2_weight * lr.

    Parameters
    ----------
    relevances : array
        Relevance logits.
    l1_weight : float
        L1 penalty strength (sparsity).
    l2_weight : float
        L2 penalty strength (shrinkage).
    lr : float
        Learning rate scaling. Default: 1.0.

    Returns
    -------
    regularized_relevances : array
        Thresholded and shrunk relevances.
    """
    alpha = l1_weight * lr
    beta = l2_weight * lr
    thresholded = jnp.sign(relevances) * jnp.maximum(
        jnp.abs(relevances) - alpha, 0.0
    )
    return thresholded / (1.0 + beta)
