"""
Riemannian Matrix Robust Soft LVQ (RiemannianMRSLVQ).

Robust soft LVQ on a Riemannian manifold with an adaptive matrix in the
tangent space. A sample :math:`x` is seen from prototype :math:`w_k`
through its tangent vector :math:`\\operatorname{Log}_{w_k}(x)`, written
in orthonormal coordinates :math:`u_k(x)` of the tangent space, and

.. math::

    d_\\Lambda^2(x, w_k) = u_k(x)^T \\Lambda\\, u_k(x), \\qquad
    \\Lambda = \\Omega \\Omega^T, \\qquad
    f_k(x) = -\\frac{d_\\Lambda^2(x, w_k)}{2\\sigma^2}.

One matrix is learned for all prototypes.

References
----------
.. [1] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
.. [2] Tang, F., Feng, H., Tino, P., Si, B., & Ji, D. (2021).
       Probabilistic learning vector quantization on manifold of
       symmetric positive definite matrices. Neural Networks, 142,
       105-118.
.. [3] Tang, F., Tino, P., & Yu, H. (2023). Generalized learning vector
       quantization with log-Euclidean metric learning on symmetric
       positive-definite manifold. IEEE Transactions on Cybernetics,
       53(8), 5178-5190.
"""

import jax
import jax.numpy as jnp

from prosemble.core.protocols import Manifold
from prosemble.models.riemannian_rslvq import RiemannianRSLVQ
from prosemble.models.riemannian_srng import _check_log_map, _tangent_coordinates
from prosemble.models.soft_mixin import SoftMetricMixin


class RiemannianMRSLVQ(SoftMetricMixin, RiemannianRSLVQ):
    """Riemannian Matrix Robust Soft LVQ.

    :class:`RiemannianRSLVQ` with a learned metric in the tangent space.
    Each component is a Gaussian in normal coordinates around its
    prototype,

    .. math::

        p(x | k) \\propto \\exp\\left(
            -\\frac{u_k(x)^T \\Lambda\\, u_k(x)}{2\\sigma^2}\\right),

    where :math:`u_k(x)` are the coordinates of
    :math:`\\operatorname{Log}_{w_k}(x)` in an orthonormal frame of the
    tangent space at :math:`w_k`. With the exact log map
    :math:`\\|u_k(x)\\|` is the geodesic distance, so
    :math:`\\Lambda = I` gives the isotropic Riemannian Gaussian, and
    training starts there. A learned
    :math:`\\Lambda` weights and correlates the directions in which a
    sample can differ from a prototype.

    The coordinates are

    - SO(n): :math:`w^T \\operatorname{Log}_w(x) / \\sqrt2`, an element of
      the Lie algebra;
    - SPD(n): :math:`\\log(w^{-1/2} x w^{-1/2})`;
    - Grassmannian: the horizontal matrix
      :math:`\\operatorname{Log}_w(x)` itself;
    - Poincare ball: :math:`\\lambda_w \\operatorname{Log}_w(x)`.

    On SO(n), SPD(n) and the Poincare ball these frames are carried from
    one prototype to another by an isometry of the manifold (left
    translation, the congruence :math:`z \\mapsto w^{1/2} z w^{1/2}`,
    and Mobius translation), so one matrix has the same meaning at every
    prototype. On the Grassmannian the coordinates are those of the
    ambient space.

    The matrix acts on the flattened coordinates, which have the size of
    a point of the manifold. Symmetric and skew-symmetric coordinates
    repeat their entries, so :math:`\\Omega` has more parameters than the
    dimension of the manifold requires.

    Metric learning for margin-based LVQ on SPD matrices with the
    log-Euclidean metric was introduced by Tang, Tino and Yu (2023). This
    model is probabilistic, measures the tangent vector at each
    prototype, and covers all four manifolds.

    Unlike the matrix variants of :class:`RiemannianSRNG`, which apply
    the matrix to the ambient tangent vector, the distance for
    :math:`\\Lambda = I` is the Riemannian length of the tangent vector
    on every manifold.

    Parameters
    ----------
    manifold : SO, SPD, Grassmannian or HyperbolicPoincare
        Riemannian manifold instance defining the geometry.
    latent_dim : int, optional
        Number of columns of :math:`\\Omega`. Default: the number of
        entries of a point of the manifold.
    matrix_lr_ratio : float
        Learning rate of the matrix relative to ``lr``. Default: 0.3.
    log_map : {'exact', 'first_order'}, optional
        Log map on SO(n) and the Grassmannian. Default: ``'exact'`` on
        SO(n), ``'first_order'`` on the Grassmannian. The SPD and
        hyperbolic log maps are always exact.
    sigma : float or 'auto'
        Initial width :math:`\\sigma` of the mixture components.
        ``'auto'`` (default) sets :math:`\\sigma^2` to the mean squared
        distance from a training sample to its closest prototype of the
        same class at initialisation.
    adapt_sigma : bool
        If True (default), learn :math:`\\sigma` by gradient descent on
        the cost. If False, keep it at its initial value.
    rejection_confidence : float, optional
        Minimum class posterior for ``predict_with_rejection``.
        Default: None (no rejection).
    tau : float
        Safety factor in (0, 1] for the geodesic step of a prototype.
        Default: 0.95.
    update_rule : {'riemannian', 'projected'}
        How prototypes are moved. Default: ``'riemannian'``.

    Attributes
    ----------
    omega_ : array of shape (n_features_flat, latent_dim)
        Learned map :math:`\\Omega`.
    sigma_ : float
        Learned width :math:`\\sigma` of the mixture.

    See Also
    --------
    RiemannianRSLVQ : The model without a matrix, with the remaining parameters.
    RiemannianSMNG : Margin-based matrix model on the ambient tangent vector.
    """

    _hyperparams = ('log_map',)
    def __init__(self, manifold: Manifold, latent_dim=None, matrix_lr_ratio=0.3,
                 log_map=None,
                 sigma='auto', adapt_sigma=True,
                 rejection_confidence=None, tau=0.95,
                 update_rule='riemannian',
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, optimizer='adam',
                 callbacks=None, use_scan=False, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            manifold=manifold, sigma=sigma, adapt_sigma=adapt_sigma,
            rejection_confidence=rejection_confidence, tau=tau,
            update_rule=update_rule,
            n_prototypes_per_class=n_prototypes_per_class,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, optimizer=optimizer,
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
        self.log_map = _check_log_map(log_map, manifold)
        self._setup_metric(latent_dim, matrix_lr_ratio)

    def _tangent_coordinates(self, X, prototypes):
        """Flattened tangent coordinates, shape (n, p, n_features_flat)."""
        n, p = X.shape[0], prototypes.shape[0]
        X_m = self._reshape_to_manifold(X, n)
        W_m = self._reshape_to_manifold(prototypes, p)

        def coordinates(w, x):
            return _tangent_coordinates(self.manifold, w, x, self.log_map)

        to_all_x = jax.vmap(coordinates, in_axes=(None, 0))
        coords = jax.vmap(to_all_x, in_axes=(0, None))(W_m, X_m)  # (p, n, ...)
        return jnp.moveaxis(coords, 0, 1).reshape(n, p, -1)

    def _squared_distances(self, params, X):
        coords = self._tangent_coordinates(X, params['prototypes'])
        projected = self._apply_metric(coords, params[self._metric_key])
        return jnp.sum(projected ** 2, axis=2)
