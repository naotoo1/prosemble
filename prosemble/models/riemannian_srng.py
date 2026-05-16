"""
Supervised Riemannian Neural Gas (RiemannianSRNG).

Extends GLVQ-style supervised classification to Riemannian manifolds
using geodesic distances and Neural Gas neighborhood cooperation.
Prototypes live on the manifold and are updated via projected gradient
descent (Euclidean gradient + manifold projection).

Supports SO(n), SPD(n), and Grassmannian(n,k) manifolds.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel, SupervisedState
from prosemble.core.activations import sigmoid_beta
from prosemble.core.manifolds import SO, SPD, Grassmannian
from prosemble.core.protocols import Manifold


# ---------------------------------------------------------------------------
# Differentiable manifold operations for autodiff-based training
# ---------------------------------------------------------------------------

def _logm_spd_diff(M):
    """Differentiable matrix log for SPD matrices via eigendecomposition.

    Unlike ``jsl.funm(A, jnp.log)`` (which uses Schur decomposition with
    no JAX differentiation rule), this uses ``jnp.linalg.eigh`` which
    has full autodiff support.
    """
    eigvals, eigvecs = jnp.linalg.eigh(M)
    eigvals = jnp.maximum(eigvals, 1e-10)
    return eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T


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
    """Differentiable tangent vector approximation on SO(n).

    Returns the skew-symmetric part of R^T S mapped to the tangent
    space at R. This is the first-order approximation of the true
    logarithmic map, exact when R and S are close.

    .. math::

        \\text{Log}_R(S) \\approx R \\cdot \\text{skew}(R^T S)
        = R \\cdot \\frac{R^T S - S^T R}{2}
    """
    RtS = R.T @ S
    skew = (RtS - RtS.T) / 2.0
    return R @ skew


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
    """
    return Q2 - Q1 @ (Q1.T @ Q2)


class RiemannianSRNG(SupervisedPrototypeModel):
    """Supervised Riemannian Neural Gas.

    Combines three key ideas:

    - GLVQ loss: :math:`(d^+ - d^-) / (d^+ + d^-)` for margin-based classification
    - Neural Gas cooperation: all same-class prototypes participate in
      the loss, weighted by rank via :math:`\\exp(-\\text{rank} / \\gamma)`
    - Geodesic distance: :math:`d(x, w)` computed via the manifold's
      intrinsic metric (matrix logarithm + Frobenius norm)

    Prototypes live on the manifold and are updated via projected gradient
    descent: optax computes Euclidean gradients, then
    :meth:`manifold.project` maps prototypes back to the manifold after
    each step.

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
    tau : float
        Injectivity radius safety factor for manifold projection.
        Default: 0.95.
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
                 gamma_decay=None, tau=0.95, n_prototypes_per_class=1,
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
        self.tau = tau
        self.gamma_ = None

        # Ensure gamma is frozen from optimizer (not trainable)
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

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
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / self.max_iter)

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
        hp['tau'] = self.tau
        # Store manifold type and params for reconstruction
        manifold = self.manifold
        hp['manifold_type'] = type(manifold).__name__
        if hasattr(manifold, 'n'):
            hp['manifold_n'] = manifold.n
        if hasattr(manifold, 'k'):
            hp['manifold_k'] = manifold.k
        return hp

    @classmethod
    def _reconstruct_manifold(cls, hp):
        """Reconstruct manifold from saved hyperparameters."""
        from prosemble.core.manifolds import SO, SPD, Grassmannian
        mtype = hp.get('manifold_type', '')
        if mtype == 'SO':
            return SO(int(hp['manifold_n']))
        elif mtype == 'SPD':
            return SPD(int(hp['manifold_n']))
        elif mtype == 'Grassmannian':
            return Grassmannian(int(hp['manifold_n']), int(hp['manifold_k']))
        else:
            raise ValueError(f"Unknown manifold type: {mtype}")

    @classmethod
    def _pre_load_construct(cls, hyperparams, metadata):
        manifold = cls._reconstruct_manifold(hyperparams)
        hyperparams.pop('manifold_type', None)
        hyperparams.pop('manifold_n', None)
        hyperparams.pop('manifold_k', None)
        hyperparams['manifold'] = manifold
        return hyperparams
