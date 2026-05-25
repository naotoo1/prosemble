"""
Riemannian manifold primitives for prototype learning.

Provides geodesic distance, exponential map, and logarithmic map
for manifolds commonly arising in machine learning:

- SO(n): Special orthogonal group (rotation matrices)
- SPD(n): Symmetric positive definite matrices
- Grassmannian Gr(n, k): k-dimensional subspaces of R^n
- HyperbolicPoincare(d): Poincare ball model of hyperbolic space

All operations are JIT-compilable and vmap-compatible.

References
----------
Schwarz, Psenickova, Villmann, Röhrbein (2026).
Topology-Preserving Prototype Learning on Riemannian Manifolds.
ESANN 2026.

Ganea, O., Becigneul, G., & Hofmann, T. (2018). Hyperbolic Neural
Networks. NeurIPS 2018.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl


# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

def logm_safe(A):
    """Matrix logarithm via funm, with numerical safety.

    Uses JAX's ``funm`` to compute logm. For complex eigenvalues
    (e.g. rotation matrices), operates in complex domain and returns
    the real part.

    Parameters
    ----------
    A : array of shape (..., n, n)

    Returns
    -------
    logA : array of shape (..., n, n)
    """
    return jsl.funm(A.astype(jnp.complex64), jnp.log).real


def sqrt_spd(A):
    """Matrix square root for symmetric positive definite matrices.

    Uses eigendecomposition: :math:`A^{1/2} = V \operatorname{diag}(\sqrt{\lambda}) V^T`.

    Parameters
    ----------
    A : array of shape (n, n), symmetric positive definite

    Returns
    -------
    A_sqrt : array of shape (n, n)
    """
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-10)
    return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T


def inv_sqrt_spd(A):
    """Inverse matrix square root for symmetric positive definite matrices.

    Uses eigendecomposition: :math:`A^{-1/2} = V \operatorname{diag}(1/\sqrt{\lambda}) V^T`.

    Parameters
    ----------
    A : array of shape (n, n), symmetric positive definite

    Returns
    -------
    A_inv_sqrt : array of shape (n, n)
    """
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-10)
    return eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals)) @ eigvecs.T


# ---------------------------------------------------------------------------
# SO(n) — Special Orthogonal Group
# ---------------------------------------------------------------------------

class SO:
    """Special orthogonal group SO(n): rotation matrices.

    Points are :math:`n \times n` orthogonal matrices with det = +1.
    Geodesic distance uses the bi-invariant metric.

    Parameters
    ----------
    n : int
        Dimension of the rotation group.
    """

    def __init__(self, n: int):
        self.n = n
        self.point_shape = (n, n)

    def distance(self, R, S):
        """Geodesic distance: d(R, S) = ||logm(R^T S)||_F / sqrt(2)."""
        RtS = R.T @ S
        log_RtS = logm_safe(RtS)
        return jnp.linalg.norm(log_RtS, 'fro') / jnp.sqrt(2.0)

    def distance_squared(self, R, S):
        """Squared geodesic distance."""
        d = self.distance(R, S)
        return d ** 2

    def log_map(self, R, S):
        """Logarithmic map: Log_R(S) = R @ logm(R^T @ S).

        Maps point S on the manifold to a tangent vector at R.
        """
        return R @ logm_safe(R.T @ S)

    def exp_map(self, R, V):
        """Exponential map: Exp_R(V) = R @ expm(R^T @ V).

        Maps tangent vector V at R back to the manifold.
        """
        return R @ jsl.expm(R.T @ V)

    def random_point(self, key):
        """Generate a random rotation matrix via QR decomposition.

        Flips only the last column to ensure det = +1, preserving
        the Haar-uniform distribution on SO(n).
        """
        A = jax.random.normal(key, (self.n, self.n))
        Q, _ = jnp.linalg.qr(A)
        # Flip only last column to ensure det = +1 (Haar measure)
        Q = Q.at[:, -1].mul(jnp.sign(jnp.linalg.det(Q)))
        return Q

    def belongs(self, R):
        """Check if R is in SO(n): :math:`R^T R \\approx I` and :math:`\\det(R) \\approx +1`."""
        ortho = jnp.allclose(R.T @ R, jnp.eye(self.n), atol=1e-4)
        det_pos = jnp.abs(jnp.linalg.det(R) - 1.0) < 1e-4
        return ortho & det_pos

    def project(self, R):
        """Project to nearest rotation matrix via polar decomposition."""
        U, P = jsl.polar(R)
        U = U * jnp.sign(jnp.linalg.det(U))
        return U

    def injectivity_radius(self, R):
        """Injectivity radius of SO(n) is :math:`\\pi`."""
        return jnp.pi


# ---------------------------------------------------------------------------
# SPD(n) — Symmetric Positive Definite Matrices
# ---------------------------------------------------------------------------

class SPD:
    """Manifold of :math:`n \\times n` symmetric positive definite matrices.

    Uses the affine-invariant Riemannian metric.

    Parameters
    ----------
    n : int
        Matrix dimension.
    """

    def __init__(self, n: int):
        self.n = n
        self.point_shape = (n, n)

    def distance(self, A, B):
        """Geodesic distance: d(A, B) = ||logm(A^{-1/2} B A^{-1/2})||_F."""
        A_isqrt = inv_sqrt_spd(A)
        M = A_isqrt @ B @ A_isqrt
        return jnp.linalg.norm(logm_safe(M), 'fro')

    def distance_squared(self, A, B):
        """Squared geodesic distance."""
        d = self.distance(A, B)
        return d ** 2

    def log_map(self, A, B):
        """Log map: Log_A(B) = A^{1/2} logm(A^{-1/2} B A^{-1/2}) A^{1/2}."""
        A_sqrt = sqrt_spd(A)
        A_isqrt = inv_sqrt_spd(A)
        M = A_isqrt @ B @ A_isqrt
        return A_sqrt @ logm_safe(M) @ A_sqrt

    def exp_map(self, A, V):
        """Exp map: Exp_A(V) = A^{1/2} expm(A^{-1/2} V A^{-1/2}) A^{1/2}."""
        A_sqrt = sqrt_spd(A)
        A_isqrt = inv_sqrt_spd(A)
        inner = A_isqrt @ V @ A_isqrt
        return A_sqrt @ jsl.expm(inner) @ A_sqrt

    def random_point(self, key):
        """Generate random SPD matrix: :math:`A = L L^T + \\epsilon I`."""
        L = jax.random.normal(key, (self.n, self.n))
        return L @ L.T + 0.1 * jnp.eye(self.n)

    def belongs(self, A):
        """Check if A is SPD: symmetric and all eigenvalues > 0."""
        sym = jnp.allclose(A, A.T, atol=1e-4)
        pos = jnp.all(jnp.linalg.eigvalsh(A) > 0)
        return sym & pos

    def project(self, A):
        """Project to nearest SPD: symmetrize and clamp eigenvalues."""
        A_sym = (A + A.T) / 2.0
        eigvals, eigvecs = jnp.linalg.eigh(A_sym)
        eigvals = jnp.maximum(eigvals, 1e-6)
        return eigvecs @ jnp.diag(eigvals) @ eigvecs.T

    def injectivity_radius(self, A):
        """SPD manifold has infinite injectivity radius."""
        return jnp.inf


# ---------------------------------------------------------------------------
# Grassmannian Gr(n, k)
# ---------------------------------------------------------------------------

class Grassmannian:
    """Grassmannian manifold Gr(n, k): k-dimensional subspaces of R^n.

    Points are represented as orthonormal bases Q of shape (n, k)
    with Q^T Q = I_k.

    Parameters
    ----------
    n : int
        Ambient dimension.
    k : int
        Subspace dimension.
    """

    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.point_shape = (n, k)

    def distance(self, Q1, Q2):
        """Geodesic distance via principal angles.

        d(Q1, Q2) = ||theta||_2 where theta = arccos(svd(Q1^T Q2)).
        """
        M = Q1.T @ Q2
        svals = jnp.linalg.svd(M, compute_uv=False)
        svals = jnp.clip(svals, -1.0, 1.0)
        angles = jnp.arccos(svals)
        return jnp.linalg.norm(angles)

    def distance_squared(self, Q1, Q2):
        """Squared geodesic distance."""
        d = self.distance(Q1, Q2)
        return d ** 2

    def log_map(self, Q1, Q2):
        """Logarithmic map on the Grassmannian.

        Computes the tangent vector at Q1 pointing toward Q2 using
        aligned principal angle decomposition (Edelman et al. 1998).

        The key insight is to align both subspaces via the SVD of
        Q1^T Q2, ensuring the principal angles and directions correspond.
        """
        M = Q1.T @ Q2
        U_m, cos_theta, V_mt = jnp.linalg.svd(M, full_matrices=False)
        cos_theta = jnp.clip(cos_theta, -1.0, 1.0)

        # Aligned perpendicular component
        N = Q2 @ V_mt.T - Q1 @ U_m * cos_theta[None, :]

        # Column norms = sin(theta_i), columns are orthogonal
        sin_theta = jnp.linalg.norm(N, axis=0)
        sin_theta_safe = jnp.maximum(sin_theta, 1e-10)

        # Principal angles
        theta = jnp.arctan2(sin_theta, cos_theta)

        # Unit perpendicular directions
        U_dirs = N / sin_theta_safe[None, :]

        # Tangent vector at Q1 (map back from aligned frame)
        return U_dirs * theta[None, :] @ U_m.T

    def exp_map(self, Q, V):
        """Exponential map on the Grassmannian.

        Maps tangent vector V at Q back to the manifold.
        """
        # SVD of tangent vector
        U, S, Vt = jnp.linalg.svd(V, full_matrices=False)
        # S contains the principal angles (theta)
        cos_S = jnp.cos(S)
        sin_S = jnp.sin(S)

        # New point: Q @ V_r @ diag(cos) + U @ diag(sin)
        return Q @ (Vt.T * cos_S[None, :]) + U * sin_S[None, :]

    def random_point(self, key):
        """Generate a random point on Gr(n, k) via QR decomposition."""
        A = jax.random.normal(key, (self.n, self.k))
        Q, _ = jnp.linalg.qr(A)
        return Q

    def belongs(self, Q):
        """Check if Q represents a point on Gr(n, k): :math:`Q^T Q \\approx I_k`."""
        return jnp.allclose(Q.T @ Q, jnp.eye(self.k), atol=1e-4)

    def project(self, Q):
        """Project to nearest orthonormal basis via QR."""
        Q_new, _ = jnp.linalg.qr(Q)
        return Q_new

    def injectivity_radius(self, Q):
        """Injectivity radius of Gr(n,k) is :math:`\\pi/2`."""
        return jnp.pi / 2.0


# ---------------------------------------------------------------------------
# HyperbolicPoincare(d) — Poincaré Ball Model of Hyperbolic Space
# ---------------------------------------------------------------------------

def _mobius_add(x, y, eps=1e-7):
    """Möbius addition in the Poincaré ball.

    .. math::

        x \\oplus y = \\frac{(1 + 2\\langle x, y\\rangle + \\|y\\|^2) x
                           + (1 - \\|x\\|^2) y}
                          {1 + 2\\langle x, y\\rangle + \\|x\\|^2 \\|y\\|^2}

    Parameters
    ----------
    x, y : arrays of shape (d,)

    Returns
    -------
    result : array of shape (d,)
    """
    x_sq = jnp.sum(x ** 2)
    y_sq = jnp.sum(y ** 2)
    xy = jnp.sum(x * y)
    num = (1.0 + 2.0 * xy + y_sq) * x + (1.0 - x_sq) * y
    denom = 1.0 + 2.0 * xy + x_sq * y_sq
    return num / (denom + eps)


def _conformal_factor(x, eps=1e-7):
    """Conformal factor :math:`\\lambda_x = 2 / (1 - \\|x\\|^2)`.

    Parameters
    ----------
    x : array of shape (d,)

    Returns
    -------
    scalar
    """
    return 2.0 / (1.0 - jnp.sum(x ** 2) + eps)


class HyperbolicPoincare:
    """Poincaré ball model of hyperbolic space :math:`\\mathbb{H}^d`.

    Points are vectors in :math:`\\mathbb{B}^d = \\{x \\in \\mathbb{R}^d : \\|x\\| < 1\\}`.
    The Riemannian metric is :math:`g_x = \\lambda_x^2 g_E` where
    :math:`\\lambda_x = 2/(1-\\|x\\|^2)` is the conformal factor.

    This manifold is the natural geometry for hierarchical and
    tree-structured data — it can embed trees with arbitrarily low
    distortion (Sarkar 2011).

    Parameters
    ----------
    d : int
        Dimension of the hyperbolic space.
    eps : float
        Numerical safety margin for boundary clamping.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        self.d = d
        self.eps = eps
        self.point_shape = (d,)

    def distance(self, x, y):
        """Geodesic distance in the Poincaré ball.

        .. math::

            d(x, y) = \\operatorname{arcosh}\\!\\left(1 + \\frac{2\\|x - y\\|^2}
                      {(1 - \\|x\\|^2)(1 - \\|y\\|^2)}\\right)
        """
        diff_sq = jnp.sum((x - y) ** 2)
        x_sq = jnp.sum(x ** 2)
        y_sq = jnp.sum(y ** 2)
        denom = (1.0 - x_sq) * (1.0 - y_sq)
        arg = 1.0 + 2.0 * diff_sq / (denom + self.eps)
        # Stable arcosh with safe gradient: arcosh(z) = log(z + sqrt(z^2-1))
        # Clamp arg >= 1 + eps to prevent gradient blow-up near z=1
        arg = jnp.maximum(arg, 1.0 + 1e-6)
        return jnp.arccosh(arg)

    def distance_squared(self, x, y):
        """Squared geodesic distance."""
        d = self.distance(x, y)
        return d ** 2

    def log_map(self, x, y):
        """Logarithmic map: tangent vector at x pointing toward y.

        .. math::

            \\text{Log}_x(y) = \\frac{2}{\\lambda_x}
            \\operatorname{arctanh}(\\|-x \\oplus y\\|)
            \\frac{-x \\oplus y}{\\|-x \\oplus y\\|}
        """
        neg_x = -x
        add_result = _mobius_add(neg_x, y, eps=self.eps)
        norm = jnp.linalg.norm(add_result)
        norm_safe = jnp.maximum(norm, 1e-10)
        # arctanh is only defined for |z| < 1, clamp
        norm_clamped = jnp.minimum(norm, 1.0 - 1e-7)
        lam_x = _conformal_factor(x, eps=self.eps)
        scale = (2.0 / lam_x) * jnp.arctanh(norm_clamped)
        return scale * add_result / norm_safe

    def exp_map(self, x, v):
        """Exponential map: move from x along tangent vector v.

        .. math::

            \\text{Exp}_x(v) = x \\oplus \\left(
            \\tanh\\!\\left(\\frac{\\lambda_x \\|v\\|}{2}\\right)
            \\frac{v}{\\|v\\|}\\right)
        """
        v_norm = jnp.linalg.norm(v)
        v_norm_safe = jnp.maximum(v_norm, 1e-10)
        lam_x = _conformal_factor(x, eps=self.eps)
        direction = v / v_norm_safe
        t = jnp.tanh(lam_x * v_norm / 2.0) * direction
        return _mobius_add(x, t, eps=self.eps)

    def random_point(self, key):
        """Generate a random point uniformly in the Poincaré ball.

        Uses the radial transform: sample direction uniformly on S^{d-1},
        then radius r ~ Beta(1, d) to get uniform distribution in B^d,
        scaled to stay safely inside the ball.
        """
        key1, key2 = jax.random.split(key)
        # Random direction on unit sphere
        direction = jax.random.normal(key1, (self.d,))
        direction = direction / (jnp.linalg.norm(direction) + 1e-10)
        # Uniform radius in ball: r^d ~ U[0,1] => r = U^{1/d}
        u = jax.random.uniform(key2, (), minval=0.01, maxval=0.95)
        r = u ** (1.0 / self.d)
        # Scale to stay inside ball with margin
        r = r * (1.0 - self.eps)
        return r * direction

    def belongs(self, x):
        """Check if x is in the Poincaré ball: :math:`\\|x\\| < 1`."""
        return jnp.sum(x ** 2) < 1.0

    def project(self, x):
        """Project to the interior of the Poincaré ball.

        Clamps the norm to be strictly less than 1.
        """
        max_norm = 1.0 - self.eps
        norm = jnp.linalg.norm(x)
        scale = jnp.minimum(max_norm / (norm + 1e-10), 1.0)
        return x * scale

    def injectivity_radius(self, x):
        """Hyperbolic space has infinite injectivity radius."""
        return jnp.inf
