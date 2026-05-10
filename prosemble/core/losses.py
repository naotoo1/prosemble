"""
Loss functions for prototype-based learning.

All loss functions are differentiable by jax.grad and JIT-compatible.
They use jnp.where masking (not boolean indexing) for d+/d- extraction.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial

from .activations import identity


# --- Helpers ---

@jit
def _get_dp_dm(distances, target_labels, prototype_labels):
    """Extract d+ (min same-class) and d- (min different-class) distances.

    Parameters
    ----------
    distances : array of shape (n, p)
        Distance from each sample to each prototype.
    target_labels : array of shape (n,)
        True labels.
    prototype_labels : array of shape (p,)
        Prototype labels.

    Returns
    -------
    dp : array of shape (n,)
        Distance to closest same-class prototype.
    dm : array of shape (n,)
        Distance to closest different-class prototype.
    """
    # same_class[i, j] = True if sample i and prototype j share label
    same_class = (target_labels[:, None] == prototype_labels[None, :])
    diff_class = ~same_class

    INF = jnp.finfo(distances.dtype).max
    dp = jnp.min(jnp.where(same_class, distances, INF), axis=1)
    dm = jnp.min(jnp.where(diff_class, distances, INF), axis=1)
    return dp, dm


@jit
def _get_dp_dm_with_indices(distances, target_labels, prototype_labels):
    """Extract d+/d- with winner indices.

    Returns
    -------
    dp, dm : arrays of shape (n,)
    wp, wm : arrays of shape (n,) — indices of winning prototypes
    """
    same_class = (target_labels[:, None] == prototype_labels[None, :])
    diff_class = ~same_class

    INF = jnp.finfo(distances.dtype).max
    d_same = jnp.where(same_class, distances, INF)
    d_diff = jnp.where(diff_class, distances, INF)

    dp = jnp.min(d_same, axis=1)
    dm = jnp.min(d_diff, axis=1)
    wp = jnp.argmin(d_same, axis=1)
    wm = jnp.argmin(d_diff, axis=1)
    return dp, dm, wp, wm


# --- GLVQ Loss ---

@jit
def glvq_loss(distances, target_labels, prototype_labels,
              margin=0.0):
    """Generalized LVQ loss.

    mu_i = (d+_i - d-_i) / (d+_i + d-_i)

    Parameters
    ----------
    distances : array of shape (n, p)
    target_labels : array of shape (n,)
    prototype_labels : array of shape (p,)
    margin : float
        Margin added to mu before transfer.

    Returns
    -------
    scalar
        Mean loss over samples.
    """
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    mu = (dp - dm) / (dp + dm + 1e-10)
    return jnp.mean(mu + margin)


def glvq_loss_with_transfer(distances, target_labels, prototype_labels,
                            transfer_fn=identity, margin=0.0, beta=10.0):
    """GLVQ loss with configurable transfer function.

    loss = mean(transfer(mu + margin, beta))

    Parameters
    ----------
    distances : array of shape (n, p)
    target_labels : array of shape (n,)
    prototype_labels : array of shape (p,)
    transfer_fn : callable
        Activation function (identity, sigmoid_beta, swish_beta).
    margin : float
    beta : float
        Transfer function parameter.

    Returns
    -------
    scalar
    """
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    mu = (dp - dm) / (dp + dm + 1e-10)
    return jnp.mean(transfer_fn(mu + margin, beta))


# --- LVQ1 / LVQ2.1 Losses ---

@jit
def lvq1_loss(distances, target_labels, prototype_labels):
    """LVQ1 loss: d+ when correct, -d- when wrong.

    Parameters
    ----------
    distances : array of shape (n, p)
    target_labels : array of shape (n,)
    prototype_labels : array of shape (p,)

    Returns
    -------
    scalar
    """
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    # When d+ < d- (correct): loss = d+ (want to minimize)
    # When d+ > d- (wrong): loss = -d- (want to push away)
    mu = jnp.where(dp <= dm, dp, -dm)
    return jnp.mean(mu)


@jit
def lvq21_loss(distances, target_labels, prototype_labels):
    """LVQ2.1 loss: d+ - d- (unnormalized).

    Parameters
    ----------
    distances : array of shape (n, p)
    target_labels : array of shape (n,)
    prototype_labels : array of shape (p,)

    Returns
    -------
    scalar
    """
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    return jnp.mean(dp - dm)


# --- Probabilistic Losses ---

@jit
def _class_probabilities(distances, target_labels, prototype_labels, sigma):
    """Compute Gaussian mixture class probabilities.

    p(k|x) = exp(-d²/2σ²) / Σexp(-d²/2σ²)
    P(class|x) = Σ_{k∈class} p(k|x)

    Returns
    -------
    whole : array (n,) — total probability
    correct : array (n,) — probability of correct class
    wrong : array (n,) — probability of wrong classes
    """
    # Compute conditional probabilities
    log_probs = -distances / (2.0 * sigma ** 2)
    # Normalize via log-sum-exp
    log_norm = jnp.max(log_probs, axis=1, keepdims=True)
    probs = jnp.exp(log_probs - log_norm)
    probs = probs / jnp.sum(probs, axis=1, keepdims=True)

    # Sum probabilities per class
    same_class = (target_labels[:, None] == prototype_labels[None, :])
    correct = jnp.sum(probs * same_class, axis=1)
    whole = jnp.sum(probs, axis=1)  # should be 1.0
    wrong = whole - correct

    return whole, correct, wrong


@jit
def nllr_loss(distances, target_labels, prototype_labels, sigma=1.0):
    """Negative Log-Likelihood Ratio loss (for SLVQ).

    loss = -log(P(correct) / P(wrong))

    Parameters
    ----------
    distances : array of shape (n, p)
    target_labels : array of shape (n,)
    prototype_labels : array of shape (p,)
    sigma : float
        Bandwidth of Gaussian mixture.

    Returns
    -------
    scalar
    """
    _, correct, wrong = _class_probabilities(
        distances, target_labels, prototype_labels, sigma
    )
    likelihood = correct / (wrong + 1e-10)
    return jnp.mean(-jnp.log(likelihood + 1e-10))


@jit
def rslvq_loss(distances, target_labels, prototype_labels, sigma=1.0):
    """Robust Soft LVQ loss (for RSLVQ).

    loss = -log(P(correct) / P(all))

    Parameters
    ----------
    distances : array of shape (n, p)
    target_labels : array of shape (n,)
    prototype_labels : array of shape (p,)
    sigma : float

    Returns
    -------
    scalar
    """
    whole, correct, _ = _class_probabilities(
        distances, target_labels, prototype_labels, sigma
    )
    likelihood = correct / (whole + 1e-10)
    return jnp.mean(-jnp.log(likelihood + 1e-10))


@jit
def ng_rslvq_loss(distances, target_labels, prototype_labels, sigma=1.0, gamma=1.0):
    """RSLVQ loss with Neural Gas rank-based neighborhood cooperation.

    Combines Gaussian mixture prototype probabilities with NG rank weights
    to create a neighborhood-cooperative probabilistic assignment:

        p(k|x) = exp(-d_k / 2σ²) / Σ exp(-d_j / 2σ²)   [Gaussian]
        h_k = exp(-rank_k / γ) / Σ exp(-rank_j / γ)      [NG weights]
        w_k = p(k|x) · h_k / Σ p(j|x) · h_j             [combined]

    The loss is -log(Σ w_k for correct class / Σ w_k for all).

    Parameters
    ----------
    distances : array of shape (n, p)
        Squared distances from samples to prototypes.
    target_labels : array of shape (n,)
        True class labels for samples.
    prototype_labels : array of shape (p,)
        Class labels assigned to prototypes.
    sigma : float
        Bandwidth of Gaussian mixture.
    gamma : float
        Neural Gas neighborhood range.

    Returns
    -------
    scalar
        Mean negative log-likelihood with NG cooperation.
    """
    # 1. Gaussian mixture probabilities (numerically stable)
    log_probs = -distances / (2.0 * sigma ** 2)
    log_norm = jnp.max(log_probs, axis=1, keepdims=True)
    probs = jnp.exp(log_probs - log_norm)
    probs = probs / (jnp.sum(probs, axis=1, keepdims=True) + 1e-10)

    # 2. NG rank-based weighting (double argsort for ranks)
    order = jnp.argsort(distances, axis=1)
    ranks = jnp.argsort(order, axis=1).astype(jnp.float32)
    h = jnp.exp(-ranks / (gamma + 1e-10))
    h = h / (jnp.sum(h, axis=1, keepdims=True) + 1e-10)

    # 3. Combined weights (Gaussian × NG), renormalized
    weighted_probs = h * probs
    weighted_probs = weighted_probs / (jnp.sum(weighted_probs, axis=1, keepdims=True) + 1e-10)

    # 4. Sum correct-class weighted probabilities
    same_class = (target_labels[:, None] == prototype_labels[None, :])
    correct = jnp.sum(weighted_probs * same_class, axis=1)

    # 5. RSLVQ objective: -log(P_correct)
    return jnp.mean(-jnp.log(correct + 1e-10))


# --- Cross-Entropy LVQ Loss ---

@partial(jit, static_argnums=(3,))
def cross_entropy_lvq_loss(distances, target_labels, prototype_labels, n_classes):
    """Cross-entropy LVQ loss (for CELVQ).

    1. Min distances per class via masking
    2. Negate to get logits (closer = higher)
    3. Cross-entropy against true labels

    Parameters
    ----------
    distances : array of shape (n, p)
    target_labels : array of shape (n,)
    prototype_labels : array of shape (p,)
    n_classes : int

    Returns
    -------
    scalar
    """
    from .pooling import stratified_min_pooling
    class_dists = stratified_min_pooling(distances, prototype_labels, n_classes)
    logits = -class_dists  # negate: smaller distance = larger logit
    # Numerically stable cross-entropy
    log_probs = jax.nn.log_softmax(logits, axis=1)
    target_one_hot = jax.nn.one_hot(target_labels, n_classes)
    return -jnp.mean(jnp.sum(target_one_hot * log_probs, axis=1))


# --- Margin Loss (for CBC) ---

@jit
def margin_loss(y_pred, y_true_one_hot, margin=0.3):
    """Margin loss for CBC.

    loss = relu(max_wrong - correct + margin)

    Parameters
    ----------
    y_pred : array of shape (n, n_classes)
        Predicted class probabilities.
    y_true_one_hot : array of shape (n, n_classes)
        One-hot encoded true labels.
    margin : float

    Returns
    -------
    scalar
    """
    correct = jnp.sum(y_true_one_hot * y_pred, axis=-1)
    wrong_max = jnp.max(y_pred - y_true_one_hot * 1e9, axis=-1)
    return jnp.mean(jax.nn.relu(wrong_max - correct + margin))


# --- Neural Gas Energy ---

@jit
def neural_gas_energy(distances, lam):
    """Neural Gas energy function.

    E = Σ_k h(rank_k, λ) * d(x, w_k)
    h(rank, λ) = exp(-rank / λ)

    Parameters
    ----------
    distances : array of shape (n, p)
    lam : float
        Neighborhood range parameter.

    Returns
    -------
    scalar
    """
    # Rank prototypes by distance for each sample
    order = jnp.argsort(distances, axis=1)
    ranks = jnp.argsort(order, axis=1).astype(jnp.float32)
    h = jnp.exp(-ranks / lam)
    return jnp.sum(h * distances)


# Need jax import for cross_entropy_lvq_loss
import jax
