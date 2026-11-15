"""
Supervised Riemannian Neural Gas (RiemannianSRNG).

Extends GLVQ-style supervised classification to Riemannian manifolds
using geodesic distances and Neural Gas neighborhood cooperation.
Prototypes live on the manifold and are updated via projected gradient
descent (Euclidean gradient + manifold projection).

Supports SO(n), SPD(n), Grassmannian(n,k), and HyperbolicPoincare(d) manifolds.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel, SupervisedState
from prosemble.core.activations import sigmoid_beta
from prosemble.core.manifolds import SO, SPD, Grassmannian, HyperbolicPoincare
from prosemble.core.protocols import Manifold
from prosemble.core.riemannian_optim import (
    riemannian_update, supports_riemannian_update,
)


# ---------------------------------------------------------------------------
# Differentiable manifold operations for autodiff-based training
# ---------------------------------------------------------------------------

@jax.custom_jvp
def _logm_spd_diff(M):
    """Differentiable matrix log for SPD matrices via eigendecomposition.

    Unlike ``jsl.funm(A, jnp.log)`` (which uses Schur decomposition with
    no JAX differentiation rule), this uses ``jnp.linalg.eigh``.

    The derivative is given by a custom JVP (Daleckii-Krein formula)
    rather than by differentiating through ``eigh``, whose JVP contains
    ``1 / (lambda_i - lambda_j)`` and yields inf/NaN for (near-)repeated
    eigenvalues, e.g. when a prototype coincides with a sample and
    ``M`` is close to the identity.
    """
    eigvals, eigvecs = jnp.linalg.eigh(M)
    eigvals = jnp.maximum(eigvals, 1e-10)
    return eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T


@_logm_spd_diff.defjvp
def _logm_spd_diff_jvp(primals, tangents):
    (M,), (dM,) = primals, tangents
    eigvals, eigvecs = jnp.linalg.eigh(M)
    eigvals = jnp.maximum(eigvals, 1e-10)
    log_eig = jnp.log(eigvals)
    out = eigvecs @ jnp.diag(log_eig) @ eigvecs.T

    # Divided differences of log: (log li - log lj) / (li - lj),
    # with the limit 1 / l on (near-)degenerate pairs.
    li, lj = eigvals[:, None], eigvals[None, :]
    diff = li - lj
    close = jnp.abs(diff) <= 1e-6 * jnp.maximum(li, lj)
    safe_diff = jnp.where(close, 1.0, diff)
    L = jnp.where(close,
                  2.0 / (li + lj),
                  (log_eig[:, None] - log_eig[None, :]) / safe_diff)

    dM_sym = (dM + dM.T) / 2.0
    dM_eig = eigvecs.T @ dM_sym @ eigvecs
    d_out = eigvecs @ (L * dM_eig) @ eigvecs.T
    return out, d_out


# Exact log maps on SO(n) and Gr(n, k) both reduce to one scalar function of
# the principal angles, applied to a symmetric matrix:
#
#     Phi(v) = theta / (sin(theta) cos(theta)),    v = cos^2(theta).
#
# On SO(n), theta is half the rotation angle, so Phi = angle / sin(angle).

#: Below this value of cos^2(theta) the pair is treated as lying on the cut
#: locus (rotation by pi, or a principal angle of pi / 2).
_CUT_LOCUS_COS2 = 1e-6
#: Above this value of cos^2(theta) the power series around theta = 0 is used.
_SERIES_COS2 = 0.98


def _angle_ratio(v):
    """:math:`\\Phi(v) = \\arccos\\sqrt{v} / \\sqrt{v (1 - v)}` for v in (0, 1].

    Equal to :math:`\\theta / (\\sin\\theta \\cos\\theta)` with
    :math:`v = \\cos^2\\theta`. :math:`\\Phi(1) = 1`, and :math:`\\Phi`
    diverges at the cut locus :math:`v = 0`, where ``v`` is clamped.
    """
    v = jnp.clip(v, _CUT_LOCUS_COS2, 1.0)
    near_one = v > _SERIES_COS2
    u = jnp.where(near_one, 1.0 - v, 0.0)
    series = 1.0 + u * (2 / 3 + u * (8 / 15 + u * (
        16 / 35 + u * (128 / 315 + u * 256 / 693))))
    safe = jnp.where(near_one, 0.5, v)
    closed = jnp.arccos(jnp.sqrt(safe)) / jnp.sqrt(safe * (1.0 - safe))
    return jnp.where(near_one, series, closed)


def _angle_ratio_prime(v):
    """Derivative of :func:`_angle_ratio`; zero where ``v`` is clamped."""
    inside = v > _CUT_LOCUS_COS2
    v = jnp.clip(v, _CUT_LOCUS_COS2, 1.0)
    near_one = v > _SERIES_COS2
    u = jnp.where(near_one, 1.0 - v, 0.0)
    series = -(2 / 3 + u * (16 / 15 + u * (48 / 35 + u * (
        512 / 315 + u * 1280 / 693))))
    safe = jnp.where(near_one, 0.5, v)
    phi = jnp.arccos(jnp.sqrt(safe)) / jnp.sqrt(safe * (1.0 - safe))
    closed = -(1.0 - (2.0 * safe - 1.0) * phi) / (2.0 * safe * (1.0 - safe))
    return jnp.where(inside, jnp.where(near_one, series, closed), 0.0)


@jax.custom_jvp
def _angle_ratio_matrix(M):
    """:func:`_angle_ratio` as a matrix function of a symmetric matrix.

    The eigenvalues of ``M`` are repeated by construction (each rotation
    plane of SO(n) contributes a pair), so the derivative is given by the
    Daleckii-Krein formula instead of by differentiating through ``eigh``.
    """
    eigvals, eigvecs = jnp.linalg.eigh((M + M.T) / 2.0)
    return (eigvecs * _angle_ratio(eigvals)[None, :]) @ eigvecs.T


@_angle_ratio_matrix.defjvp
def _angle_ratio_matrix_jvp(primals, tangents):
    (M,), (dM,) = primals, tangents
    eigvals, eigvecs = jnp.linalg.eigh((M + M.T) / 2.0)
    values = _angle_ratio(eigvals)
    out = (eigvecs * values[None, :]) @ eigvecs.T

    # Divided differences, with the derivative on (near-)degenerate pairs.
    li, lj = eigvals[:, None], eigvals[None, :]
    diff = li - lj
    close = jnp.abs(diff) <= 1e-3
    safe_diff = jnp.where(close, 1.0, diff)
    L = jnp.where(close,
                  _angle_ratio_prime((li + lj) / 2.0),
                  (values[:, None] - values[None, :]) / safe_diff)

    dM_sym = (dM + dM.T) / 2.0
    d_out = eigvecs @ (L * (eigvecs.T @ dM_sym @ eigvecs)) @ eigvecs.T
    return out, d_out


def _so_chordal_distance_squared(R, S):
    """Chordal (Frobenius) distance squared on SO(n).

    .. math::

        d^2(R, S) = \\|R - S\\|_F^2

    This is a standard differentiable proxy for the geodesic distance
    on SO(n), widely used in rotation averaging (Hartley et al., 2013).
    It is monotonically related to the geodesic distance for small angles.
    """
    diff = R - S
    return jnp.sum(diff ** 2)


def _spd_distance_squared_diff(A, B):
    """Differentiable squared geodesic distance on SPD(n).

    .. math::

        d^2(A, B) = \\|\\log(A^{-1/2} B A^{-1/2})\\|_F^2

    Uses eigendecomposition-based logm (differentiable) instead of
    ``jsl.funm`` (not differentiable).
    """
    from prosemble.core.manifolds import inv_sqrt_spd
    A_isqrt = inv_sqrt_spd(A)
    M = A_isqrt @ B @ A_isqrt
    logM = _logm_spd_diff(M)
    return jnp.sum(logM ** 2)


def _grassmannian_distance_squared_diff(Q1, Q2):
    """Differentiable squared geodesic distance on Gr(n,k).

    Uses SVD + arccos, both of which have JAX differentiation rules.
    """
    M = Q1.T @ Q2
    svals = jnp.linalg.svd(M, compute_uv=False)
    svals = jnp.clip(svals, -1.0 + 1e-7, 1.0 - 1e-7)
    angles = jnp.arccos(svals)
    return jnp.sum(angles ** 2)


def _so_log_map_diff(R, S):
    """First-order approximation of the log map on SO(n).

    Returns the skew-symmetric part of R^T S mapped to the tangent
    space at R.

    .. math::

        \\text{Log}_R(S) \\approx R \\cdot \\text{skew}(R^T S)
        = R \\cdot \\frac{R^T S - S^T R}{2}

    The length of this vector is proportional to :math:`\\sin\\theta` for
    a rotation angle :math:`\\theta`, not to :math:`\\theta`: it is
    accurate for small angles, largest at 90 degrees and zero at 180
    degrees, so distant rotations appear close. Use
    :func:`_so_log_map_exact` unless this form is required.
    """
    RtS = R.T @ S
    skew = (RtS - RtS.T) / 2.0
    return R @ skew


def _so_log_map_exact(R, S):
    """Log map on SO(n), differentiable.

    .. math::

        \\text{Log}_R(S) = R \\log(R^T S), \\qquad
        \\log Q = \\tfrac12 \\left(K \\, \\Phi(H) + \\Phi(H) \\, K\\right),

    with :math:`K = (Q - Q^T) / 2`, :math:`H = (2 I + Q + Q^T) / 4` and
    :math:`\\Phi` from :func:`_angle_ratio`. In each rotation plane
    :math:`K` has magnitude :math:`\\sin\\theta` and :math:`H` has the
    eigenvalue :math:`\\cos^2(\\theta / 2)`, on which
    :math:`\\Phi = \\theta / \\sin\\theta`. For n = 2 and n = 3 there is
    a single plane and :math:`\\Phi(H)` reduces to a scalar function of
    the trace.

    The log map is discontinuous at the cut locus, a rotation by
    :math:`\\pi`. Within about 2e-3 rad of it the vector is scaled down
    continuously to zero instead.
    """
    n = R.shape[-1]
    Q = R.T @ S
    K = (Q - Q.T) / 2.0
    if n in (2, 3):
        cos_angle = jnp.trace(Q) / 2.0 if n == 2 else (jnp.trace(Q) - 1.0) / 2.0
        return R @ (_angle_ratio((1.0 + cos_angle) / 2.0) * K)
    H = (2.0 * jnp.eye(n, dtype=Q.dtype) + Q + Q.T) / 4.0
    G = _angle_ratio_matrix(H)
    return R @ ((K @ G + G @ K) / 2.0)


def _spd_log_map_diff(A, B):
    """Differentiable log map on SPD(n) via eigendecomposition.

    .. math::

        \\text{Log}_A(B) = A^{1/2} \\log(A^{-1/2} B A^{-1/2}) A^{1/2}
    """
    from prosemble.core.manifolds import sqrt_spd, inv_sqrt_spd
    A_sqrt = sqrt_spd(A)
    A_isqrt = inv_sqrt_spd(A)
    M = A_isqrt @ B @ A_isqrt
    return A_sqrt @ _logm_spd_diff(M) @ A_sqrt


def _grassmannian_log_map_diff(Q1, Q2):
    """Differentiable log map on Grassmannian with safe gradient.

    The standard log map has a gradient singularity when Q1 and Q2 span
    identical or nearly identical subspaces (sin(theta) -> 0). This
    version uses the tangent space projection which is always well-conditioned:

    .. math::

        \\text{Log}_{Q_1}(Q_2) \\approx Q_2 - Q_1 (Q_1^T Q_2)

    This is the orthogonal projection of Q2 onto the normal space of Q1's
    column span, which equals the exact log map to first order and is
    smooth everywhere.

    Its singular values are :math:`\\sin\\theta_i` for the principal
    angles :math:`\\theta_i`, and it depends on the basis chosen for
    ``Q2``: another basis of the same subspace gives a different vector.
    Use :func:`_grassmannian_log_map_exact` unless this form is required.
    """
    return Q2 - Q1 @ (Q1.T @ Q2)


def _grassmannian_log_map_exact(Q1, Q2):
    """Log map on the Grassmannian, differentiable.

    With :math:`A = Q_1^T Q_2` and :math:`B = (I - Q_1 Q_1^T) Q_2`,

    .. math::

        \\text{Log}_{Q_1}(Q_2) = B A^T \\, \\Phi(A A^T),

    with :math:`\\Phi` from :func:`_angle_ratio`. Writing the principal
    angles as :math:`A = U \\cos\\Theta V^T`, :math:`B = W \\sin\\Theta V^T`,
    this equals :math:`W \\Theta U^T`. It depends only on the subspace
    spanned by ``Q2``, not on its basis.

    The log map is discontinuous at the cut locus, a principal angle of
    :math:`\\pi / 2`. Within about 1e-3 rad of it the vector is scaled
    down continuously to zero instead.
    """
    A = Q1.T @ Q2
    B = Q2 - Q1 @ A
    return B @ A.T @ _angle_ratio_matrix(A @ A.T)


def _hyperbolic_distance_squared_diff(x, y, eps=1e-5):
    """Differentiable squared geodesic distance on the Poincare ball.

    .. math::

        d^2(x, y) = \\left(\\text{arcosh}\\left(1 + \\frac{2\\|x - y\\|^2}
        {(1 - \\|x\\|^2)(1 - \\|y\\|^2)}\\right)\\right)^2

    Parameters
    ----------
    x : array of shape (d,)
        Point in the Poincare ball.
    y : array of shape (d,)
        Point in the Poincare ball.
    eps : float
        Numerical stability constant.

    Returns
    -------
    float
        Squared geodesic distance.
    """
    diff_sq = jnp.sum((x - y) ** 2)
    x_sq = jnp.sum(x ** 2)
    y_sq = jnp.sum(y ** 2)
    denom = (1.0 - x_sq) * (1.0 - y_sq)
    arg = 1.0 + 2.0 * diff_sq / (denom + eps)
    # Clamp arg >= 1 + eps to prevent gradient blow-up at z=1
    # where d/dz arcosh(z) = 1/sqrt(z^2 - 1) diverges
    arg = jnp.maximum(arg, 1.0 + 1e-6)
    dist = jnp.arccosh(arg)
    return dist ** 2


def _hyperbolic_log_map_diff(x, y, eps=1e-5):
    """Differentiable log map on the Poincare ball.

    .. math::

        \\text{Log}_x(y) = \\frac{2}{\\lambda_x} \\text{arctanh}(\\|-x \\oplus y\\|)
        \\cdot \\frac{-x \\oplus y}{\\|-x \\oplus y\\|}

    where :math:`\\lambda_x = 2 / (1 - \\|x\\|^2)` is the conformal factor
    and :math:`\\oplus` is Mobius addition.

    Parameters
    ----------
    x : array of shape (d,)
        Base point in the Poincare ball.
    y : array of shape (d,)
        Target point in the Poincare ball.
    eps : float
        Numerical stability constant.

    Returns
    -------
    array of shape (d,)
        Tangent vector at x pointing toward y.
    """
    from prosemble.core.manifolds import _mobius_add, _conformal_factor
    neg_x = -x
    add_result = _mobius_add(neg_x, y, eps=eps)
    # Gradient-safe norm: jnp.linalg.norm has 0/0 gradient at zero
    norm_sq = jnp.sum(add_result ** 2)
    norm = jnp.sqrt(norm_sq + 1e-16)
    # Clamp norm for arctanh domain (-1, 1)
    norm_clamped = jnp.minimum(norm, 1.0 - 1e-7)
    lam_x = _conformal_factor(x, eps=eps)
    # arctanh(t)/t → 1 as t → 0; use this limit to avoid 0/0 gradient
    coeff = jnp.where(norm > 1e-7, jnp.arctanh(norm_clamped) / norm, 1.0)
    return (2.0 / lam_x) * coeff * add_result


#: Accepted values of the ``log_map`` option of the tangent-space models.
LOG_MAPS = ('exact', 'first_order')


def _check_log_map(log_map, manifold):
    """Validate ``log_map`` and resolve its manifold-dependent default.

    The exact log map is the default on SO(n). On the Grassmannian the
    default stays first order: the exact map cannot be exported to ONNX
    there, and it did not improve accuracy for every model.
    """
    if log_map is None:
        return 'first_order' if isinstance(manifold, Grassmannian) else 'exact'
    if log_map not in LOG_MAPS:
        raise ValueError(
            f"log_map must be one of {LOG_MAPS} or None, got {log_map!r}"
        )
    return log_map


def _tangent_vector(manifold, w, x, log_map='exact'):
    """Differentiable tangent vector at prototype ``w`` pointing to ``x``.

    ``log_map`` selects between the exact log map and its first-order
    approximation on SO(n) and the Grassmannian. The SPD and hyperbolic
    log maps are exact in either case.
    """
    exact = log_map == 'exact'
    if isinstance(manifold, SO):
        return _so_log_map_exact(w, x) if exact else _so_log_map_diff(w, x)
    elif isinstance(manifold, SPD):
        return _spd_log_map_diff(w, x)
    elif isinstance(manifold, Grassmannian):
        if exact:
            return _grassmannian_log_map_exact(w, x)
        return _grassmannian_log_map_diff(w, x)
    elif isinstance(manifold, HyperbolicPoincare):
        return _hyperbolic_log_map_diff(w, x, eps=manifold.eps)
    return manifold.log_map(w, x)


def _tangent_coordinates(manifold, w, x, log_map='exact'):
    """Orthonormal coordinates of the tangent vector at ``w`` pointing to ``x``.

    The coordinates :math:`u` of :math:`\\xi = \\operatorname{Log}_w(x)`
    in the orthonormal tangent frame of the manifold, so that
    :math:`\\|u\\|^2 = \\|\\xi\\|_w^2` is the squared Riemannian norm:
    :math:`w^T \\xi / \\sqrt2` on SO(n),
    :math:`w^{-1/2} \\xi w^{-1/2} = \\log(w^{-1/2} x w^{-1/2})` on SPD(n),
    :math:`\\xi` on the Grassmannian and :math:`\\lambda_w \\xi` in the
    Poincare ball. They have the shape of a point of the manifold.
    """
    if isinstance(manifold, SPD):
        from prosemble.core.manifolds import inv_sqrt_spd
        w_isqrt = inv_sqrt_spd(w)
        return _logm_spd_diff(w_isqrt @ x @ w_isqrt)
    xi = _tangent_vector(manifold, w, x, log_map)
    if isinstance(manifold, SO):
        return (w.T @ xi) / jnp.sqrt(2.0)
    if isinstance(manifold, HyperbolicPoincare):
        from prosemble.core.manifolds import _conformal_factor
        return _conformal_factor(w, eps=manifold.eps) * xi
    return xi


class RiemannianSRNG(SupervisedPrototypeModel):
    """Supervised Riemannian Neural Gas.

    Combines three key ideas:

    - GLVQ loss: :math:`(d^+ - d^-) / (d^+ + d^-)` for margin-based classification
    - Neural Gas cooperation: all same-class prototypes participate in
      the loss, weighted by rank via :math:`\\exp(-\\text{rank} / \\gamma)`
    - Geodesic distance: :math:`d(x, w)` computed via the manifold's
      intrinsic metric (matrix logarithm + Frobenius norm)

    Prototypes live on the manifold and are updated intrinsically
    (``update_rule='riemannian'``). The Euclidean gradient is converted
    to the Riemannian gradient and expressed in an orthonormal frame of
    the tangent space, the optimizer computes its step there, and the
    prototype follows the geodesic:

    .. math::

        w \\leftarrow \\operatorname{Exp}_w(v), \\qquad
        \\|v\\|_w \\le \\tau \\cdot \\operatorname{inj}(w).

    The step length is therefore a geodesic length, independent of how
    the manifold is embedded, and the prototype never leaves the
    manifold. With ``optimizer='sgd'`` this is Riemannian gradient
    descent. ``update_rule='projected'`` instead takes a Euclidean step
    and maps the result back with :meth:`manifold.project`.

    The neighborhood range :math:`\\gamma` decays during training from
    :math:`\\gamma_{\\text{init}}` to :math:`\\gamma_{\\text{final}}`.
    When :math:`\\gamma \\to 0`, RiemannianSRNG recovers a Riemannian GLVQ.

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance defining the geometry.
    beta : float
        Transfer function steepness parameter for sigmoid shaping.
    gamma_init : float, optional
        Initial neighborhood range for NG cooperation.
        Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay factor for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
    lr_ratio : float
        Ratio of wrong-class to correct-class learning rate (ε⁻/ε⁺).
        Default: 0.5.
    tau : float
        Safety factor in (0, 1]. With ``update_rule='riemannian'`` the
        geodesic length of a prototype step is limited to ``tau`` times
        the injectivity radius of the manifold. Default: 0.95.
    update_rule : {'riemannian', 'projected'}
        How prototypes are moved. ``'riemannian'`` (default) takes the
        optimizer step in the tangent space, using the Riemannian
        gradient, and follows the geodesic with the exponential map.
        ``'projected'`` takes a Euclidean step and projects the result
        back onto the manifold.
    n_prototypes_per_class : int
        Number of prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    epsilon : float
        Convergence threshold on loss change.
    random_seed : int
        Random seed for reproducibility.
    optimizer : str or optax optimizer, optional
        Optimizer name ('adam', 'sgd') or optax GradientTransformation.
        Default: 'adam'.
    transfer_fn : callable, optional
        Transfer function for loss shaping (default: identity).
    margin : float
        Margin for loss computation.
    callbacks : list, optional
        List of Callback objects.
    use_scan : bool
        If True, use jax.lax.scan for training (faster, JIT-compiled).
        If False (default), use a Python for-loop with true early stopping.
    batch_size : int, optional
        Mini-batch size. If None (default), use full-batch training.
    lr_scheduler : str or optax.Schedule, optional
        Learning rate schedule. Default: None.
    lr_scheduler_kwargs : dict, optional
        Keyword arguments for the learning rate scheduler. Default: None.
    prototypes_initializer : str or callable, optional
        How to initialize prototypes. Default: 'stratified_random'.
    patience : int, optional
        Number of consecutive epochs with no improvement before stopping.
        Default: None.
    restore_best : bool
        If True, restore parameters that achieved the lowest loss. Default: False.
    class_weight : dict or 'balanced', optional
        Weights for each class. Default: None (uniform).
    gradient_accumulation_steps : int, optional
        Accumulate gradients over this many steps. Default: None.
    ema_decay : float, optional
        Exponential moving average decay for parameters. Default: None.
    freeze_params : list of str, optional
        List of parameter group names to freeze. Default: None.
    lookahead : dict, optional
        Enable lookahead optimizer wrapper. Default: None.
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training. Default: None.
    """

    def __init__(self, manifold: Manifold, beta=10.0, gamma_init=None, gamma_final=0.01,
                 gamma_decay=None, lr_ratio=0.5, tau=0.95,
                 update_rule='riemannian', n_prototypes_per_class=1,
                 max_iter=100, lr=0.01, epsilon=1e-6, random_seed=42,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=False, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            n_prototypes_per_class=n_prototypes_per_class,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, distance_fn=None,
            optimizer=optimizer, transfer_fn=transfer_fn, margin=margin,
            callbacks=callbacks, use_scan=use_scan, batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            prototypes_initializer=prototypes_initializer,
            patience=patience, restore_best=restore_best,
            class_weight=class_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ema_decay=ema_decay, freeze_params=freeze_params,
            lookahead=lookahead, mixed_precision=mixed_precision,
        )
        self.manifold = manifold
        self.beta = beta
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.lr_ratio = lr_ratio
        self.tau = tau
        self.update_rule = update_rule
        self.gamma_ = None

        if self._resolve_update_rule() == 'riemannian':
            self._optimizer = riemannian_update(
                self._optimizer, self.manifold, tau=self.tau,
            )
            self._active_optimizer = self._optimizer

        # Ensure gamma is frozen from optimizer (not trainable)
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _resolve_update_rule(self):
        """Validate ``update_rule`` and ``tau``; return the rule in effect."""
        if self.update_rule not in ('riemannian', 'projected'):
            raise ValueError(
                "update_rule must be 'riemannian' or 'projected', "
                f"got {self.update_rule!r}"
            )
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.update_rule == 'riemannian' and not supports_riemannian_update(
                self.manifold):
            warnings.warn(
                f"{type(self.manifold).__name__} does not provide "
                "gradient_to_frame, frame_to_tangent and tangent_norm; "
                "falling back to update_rule='projected'.",
                stacklevel=3,
            )
            return 'projected'
        return self.update_rule

    def _reshape_to_manifold(self, flat, n_points):
        """Reshape flat array to manifold point shape.

        Parameters
        ----------
        flat : array of shape (n_points, d_flat)
        n_points : int

        Returns
        -------
        array of shape (n_points, *point_shape)
        """
        return flat.reshape(n_points, *self.manifold.point_shape)

    def _diff_distance_squared(self, x, w):
        """Differentiable squared distance for a single pair of points.

        Dispatches to the appropriate differentiable distance based on
        manifold type. Used during training (autodiff). For inference,
        the exact geodesic distance can also be used.
        """
        if isinstance(self.manifold, SO):
            return _so_chordal_distance_squared(x, w)
        elif isinstance(self.manifold, SPD):
            return _spd_distance_squared_diff(x, w)
        elif isinstance(self.manifold, Grassmannian):
            return _grassmannian_distance_squared_diff(x, w)
        elif isinstance(self.manifold, HyperbolicPoincare):
            return _hyperbolic_distance_squared_diff(x, w, eps=self.manifold.eps)
        else:
            return self.manifold.distance_squared(x, w)

    def _geodesic_distances(self, X_manifold, W_manifold):
        """Compute pairwise squared distance matrix (differentiable).

        Parameters
        ----------
        X_manifold : array of shape (n_samples, *point_shape)
        W_manifold : array of shape (n_prototypes, *point_shape)

        Returns
        -------
        distances : array of shape (n_samples, n_prototypes)
        """
        dist_to_all = jax.vmap(self._diff_distance_squared, in_axes=(None, 0))
        dist_matrix = jax.vmap(dist_to_all, in_axes=(0, None))
        return dist_matrix(X_manifold, W_manifold)

    def _get_resume_params(self, params):
        gamma = params.get('gamma', jnp.array(self.gamma_final))
        return {
            'prototypes': params['prototypes'],
            'gamma': gamma,
        }

    def _init_state(self, X, y, key):
        key1, key2 = jax.random.split(key)
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )

        # Project initial prototypes to manifold
        n_protos = prototypes.shape[0]
        protos_manifold = self._reshape_to_manifold(prototypes, n_protos)
        protos_manifold = jax.vmap(self.manifold.project)(protos_manifold)
        prototypes = protos_manifold.reshape(n_protos, -1)

        # Compute gamma_init from prototype count if not set
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)
        gamma_init = self.gamma_init if self.gamma_init is not None else max_per_class / 2.0
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        # Compute decay factor
        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            if self.batch_size is not None:
                steps_per_epoch = (X.shape[0] + self.batch_size - 1) // self.batch_size
            else:
                steps_per_epoch = 1
            total_steps = self.max_iter * steps_per_epoch
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / total_steps)

        params = {
            'prototypes': prototypes,
            'gamma': jnp.array(gamma_init, dtype=jnp.float32),
        }
        opt_state = self._optimizer.init(params)
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        gamma = params['gamma']

        # Reshape to manifold
        n = X.shape[0]
        p = prototypes.shape[0]
        X_m = self._reshape_to_manifold(X, n)
        W_m = self._reshape_to_manifold(prototypes, p)

        # 1. Geodesic distance matrix
        distances = self._geodesic_distances(X_m, W_m)  # (n, p)

        # 2. Compute ranks within same-class prototypes
        same_class = (y[:, None] == proto_labels[None, :])  # (n, p)
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)

        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)

        # 3. Neighborhood function h = exp(-rank / gamma)
        h = jnp.exp(-ranks / (gamma + 1e-10))
        h = jnp.where(same_class, h, 0.0)

        # 4. Normalize per sample
        C = jnp.sum(h, axis=1, keepdims=True)
        h_normalized = h / (C + 1e-10)

        # 5. Closest different-class prototype distance
        d_diff = jnp.where(~same_class, distances, INF)
        dm = jnp.min(d_diff, axis=1)

        # Separate learning rates (Hammer et al. 2003: epsilon^- = lr_ratio * epsilon^+)
        # Scale gradient through dm by lr_ratio; forward pass unchanged.
        dm = jax.lax.stop_gradient(dm) + self.lr_ratio * (
            dm - jax.lax.stop_gradient(dm))

        # 6. GLVQ mu
        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)

        # 7. Transfer function
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)

        # 8. Rank-weighted sum
        weighted_cost = jnp.sum(h_normalized * cost, axis=1)
        return jnp.mean(weighted_cost)

    def _post_update(self, params):
        # Decay gamma
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)

        # Project prototypes back to manifold
        prototypes = params['prototypes']
        n_protos = prototypes.shape[0]
        protos_manifold = self._reshape_to_manifold(prototypes, n_protos)
        protos_manifold = jax.vmap(self.manifold.project)(protos_manifold)
        prototypes = protos_manifold.reshape(n_protos, -1)

        return {**params, 'gamma': new_gamma, 'prototypes': prototypes}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.gamma_ = float(params['gamma'])

    def predict(self, X):
        """Predict class labels using geodesic distance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_flat)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        n = X.shape[0]
        p = self.prototypes_.shape[0]
        X_m = self._reshape_to_manifold(X, n)
        W_m = self._reshape_to_manifold(self.prototypes_, p)
        distances = self._geodesic_distances(X_m, W_m)
        from prosemble.core.competitions import wtac
        return wtac(distances, self.prototype_labels_)

    def _get_quantizable_attrs(self):
        return {'prototypes_': self.prototypes_}

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        hp['lr_ratio'] = self.lr_ratio
        hp['tau'] = self.tau
        hp['update_rule'] = self.update_rule
        # Store manifold type and params for reconstruction
        manifold = self.manifold
        hp['manifold_type'] = type(manifold).__name__
        if hasattr(manifold, 'n'):
            hp['manifold_n'] = manifold.n
        if hasattr(manifold, 'k'):
            hp['manifold_k'] = manifold.k
        if hasattr(manifold, 'd'):
            hp['manifold_d'] = manifold.d
        return hp

    @classmethod
    def _reconstruct_manifold(cls, hp):
        """Reconstruct manifold from saved hyperparameters."""
        from prosemble.core.manifolds import SO, SPD, Grassmannian, HyperbolicPoincare
        mtype = hp.get('manifold_type', '')
        if mtype == 'SO':
            return SO(int(hp['manifold_n']))
        elif mtype == 'SPD':
            return SPD(int(hp['manifold_n']))
        elif mtype == 'Grassmannian':
            return Grassmannian(int(hp['manifold_n']), int(hp['manifold_k']))
        elif mtype == 'HyperbolicPoincare':
            return HyperbolicPoincare(int(hp['manifold_d']))
        else:
            raise ValueError(f"Unknown manifold type: {mtype}")

    @classmethod
    def _pre_load_construct(cls, hyperparams, metadata):
        manifold = cls._reconstruct_manifold(hyperparams)
        # Files written before update_rule existed were trained with
        # projected updates.
        hyperparams.setdefault('update_rule', 'projected')
        # Likewise for the log map of the tangent-space models.
        if getattr(cls, '_uses_log_map', False):
            hyperparams.setdefault('log_map', 'first_order')
        hyperparams.pop('manifold_type', None)
        hyperparams.pop('manifold_n', None)
        hyperparams.pop('manifold_k', None)
        hyperparams.pop('manifold_d', None)
        hyperparams['manifold'] = manifold
        return hyperparams
