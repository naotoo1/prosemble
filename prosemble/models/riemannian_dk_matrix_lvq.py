"""
Riemannian Differentiating Kernel GMLVQ (RiemannianDKGMLVQ).

Combines Riemannian manifold geometry with the exponential kernel distance
applied in tangent space via a learned transformation matrix
:math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T`:

.. math::

    d_\\kappa^2(x, w_k) = \\exp(v^T \\hat\\Lambda v) - 1

where :math:`v = \\text{Log}_{w_k}(x)_{\\text{flat}}` is the tangent vector.
In tangent space at :math:`w_k`, the prototype maps to the zero vector,
simplifying the full exponential kernel formula.

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.riemannian_srng import RiemannianSRNG
from prosemble.models.prototype_base import SupervisedState
from prosemble.core.activations import sigmoid_beta
from prosemble.core.initializers import identity_omega_init
from prosemble.core.competitions import wtac
from prosemble.core.manifolds import SO, SPD, Grassmannian, HyperbolicPoincare
from prosemble.models.riemannian_srng import (
    _so_log_map_diff, _spd_log_map_diff, _grassmannian_log_map_diff,
    _hyperbolic_log_map_diff,
)


class RiemannianDKGMLVQ(RiemannianSRNG):
    """Riemannian Differentiating Kernel GMLVQ.

    Extends RiemannianSRNG with exponential kernel distance applied in
    tangent space. A global transformation matrix :math:`\\hat\\Omega`
    (d_flat x latent_dim) is learned such that:

    .. math::

        \\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T

    .. math::

        d_\\kappa^2(x, w_k) = \\exp(v^T \\hat\\Lambda v) - 1

    where :math:`v = \\text{Log}_{w_k}(x)_{\\text{flat}}`. Since the
    prototype maps to the zero vector in its own tangent space, the
    exponential kernel simplifies from the full three-term formula to
    :math:`\\exp(v^T \\hat\\Lambda v) - 1`.

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance defining the geometry.
    latent_dim : int, optional
        Dimensionality of the transformation. If None, uses d_flat.
    omega_hat_scale : float
        Scale factor for omega_hat initialization. Default: 0.1.
        Smaller values prevent exp overflow at initialization.
    beta : float
        Transfer function steepness parameter.
    gamma_init : float, optional
        Initial neighborhood range for NG cooperation.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay factor for gamma.
    lr_ratio : float
        Ratio of wrong-class to correct-class learning rate (ε⁻/ε⁺).
        Default: 0.5.
    tau : float
        Injectivity radius safety factor. Default: 0.95.
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
        Default: 'adam'.
    transfer_fn : callable, optional
        Transfer function for loss shaping.
    margin : float
        Margin for loss computation.
    callbacks : list, optional
        List of Callback objects.
    use_scan : bool
        If True, use jax.lax.scan. Default: False.
    batch_size : int, optional
        Mini-batch size.
    lr_scheduler : str or optax.Schedule, optional
        Learning rate schedule.
    lr_scheduler_kwargs : dict, optional
        Keyword arguments for the learning rate scheduler.
    prototypes_initializer : str or callable, optional
        How to initialize prototypes.
    patience : int, optional
        Epochs with no improvement before stopping.
    restore_best : bool
        Restore best parameters. Default: False.
    class_weight : dict or 'balanced', optional
        Class weights.
    gradient_accumulation_steps : int, optional
        Gradient accumulation steps.
    ema_decay : float, optional
        EMA decay for parameters.
    freeze_params : list of str, optional
        Parameter groups to freeze.
    lookahead : dict, optional
        Lookahead optimizer config.
    mixed_precision : str or None, optional
        Mixed precision dtype.

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.

    See Also
    --------
    RiemannianSRNG : Base Riemannian supervised Neural Gas.
    RiemannianDKGLVQ : Gaussian kernel variant.
    RiemannianDKGRLVQ : Relevance-weighted kernel variant.
    """

    def __init__(self, manifold, latent_dim=None, omega_hat_scale=0.1,
                 beta=10.0, gamma_init=None, gamma_final=0.01,
                 gamma_decay=None, tau=0.95, lr_ratio=0.5,
                 n_prototypes_per_class=1,
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
            manifold=manifold, beta=beta,
            gamma_init=gamma_init, gamma_final=gamma_final,
            gamma_decay=gamma_decay, tau=tau, lr_ratio=lr_ratio,
            n_prototypes_per_class=n_prototypes_per_class,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, optimizer=optimizer,
            transfer_fn=transfer_fn, margin=margin,
            callbacks=callbacks, use_scan=use_scan,
            batch_size=batch_size, lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            prototypes_initializer=prototypes_initializer,
            patience=patience, restore_best=restore_best,
            class_weight=class_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ema_decay=ema_decay, freeze_params=freeze_params,
            lookahead=lookahead, mixed_precision=mixed_precision,
        )
        self.latent_dim = latent_dim
        self.omega_hat_scale = omega_hat_scale
        self.omega_hat_ = None

    def _diff_log_map(self, w, x):
        """Differentiable log map for a single (base, target) pair."""
        if isinstance(self.manifold, SO):
            return _so_log_map_diff(w, x)
        elif isinstance(self.manifold, SPD):
            return _spd_log_map_diff(w, x)
        elif isinstance(self.manifold, Grassmannian):
            return _grassmannian_log_map_diff(w, x)
        elif isinstance(self.manifold, HyperbolicPoincare):
            return _hyperbolic_log_map_diff(w, x, eps=self.manifold.eps)
        else:
            return self.manifold.log_map(w, x)

    def _compute_tangent_vectors(self, X_manifold, W_manifold):
        """Compute tangent vectors from prototypes to data via log map.

        Parameters
        ----------
        X_manifold : array of shape (n_samples, *point_shape)
        W_manifold : array of shape (n_prototypes, *point_shape)

        Returns
        -------
        tangents_flat : array of shape (n_samples, n_prototypes, d_flat)
        """
        log_to_all_x = jax.vmap(self._diff_log_map, in_axes=(None, 0))
        log_matrix = jax.vmap(log_to_all_x, in_axes=(0, None))
        tangents = log_matrix(W_manifold, X_manifold)
        tangents = jnp.moveaxis(tangents, 0, 1)
        n = X_manifold.shape[0]
        p = W_manifold.shape[0]
        return tangents.reshape(n, p, -1)

    def _get_resume_params(self, params):
        base = super()._get_resume_params(params)
        base['omega_hat'] = self.omega_hat_
        return base

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)

        d_flat = X.shape[1]
        latent_dim = self.latent_dim if self.latent_dim is not None else d_flat
        omega_hat = self.omega_hat_scale * identity_omega_init(
            d_flat, latent_dim
        )

        params = {**params, 'omega_hat': omega_hat}

        opt_state = self._optimizer.init(params)
        state = SupervisedState(
            prototypes=params['prototypes'],
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        gamma = params['gamma']
        omega_hat = params['omega_hat']
        lambda_hat = jnp.dot(omega_hat, omega_hat.T)  # (d_flat, d_flat)

        n = X.shape[0]
        p = prototypes.shape[0]
        X_m = self._reshape_to_manifold(X, n)
        W_m = self._reshape_to_manifold(prototypes, p)

        # 1. Tangent vectors via log map
        tangent_flat = self._compute_tangent_vectors(X_m, W_m)  # (n, p, d_flat)

        # 2. Exponential kernel in tangent space
        # v^T Lambda_hat v for each (sample, proto) pair
        Lv = jnp.einsum('npd,de->npe', tangent_flat, lambda_hat)  # (n, p, d_flat)
        vLv = jnp.sum(tangent_flat * Lv, axis=2)  # (n, p)
        vLv = jnp.clip(vLv, None, 20.0)  # prevent exp overflow
        distances = jnp.maximum(jnp.exp(vLv) - 1.0, 0.0)

        # 3. Compute ranks within same-class prototypes
        same_class = (y[:, None] == proto_labels[None, :])
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)

        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)

        # 4. Neighborhood function h = exp(-rank / gamma)
        h = jnp.exp(-ranks / (gamma + 1e-10))
        h = jnp.where(same_class, h, 0.0)

        # 5. Normalize per sample
        C = jnp.sum(h, axis=1, keepdims=True)
        h_normalized = h / (C + 1e-10)

        # 6. Closest different-class prototype distance
        d_diff = jnp.where(~same_class, distances, INF)
        dm = jnp.min(d_diff, axis=1)

        # 7. GLVQ mu
        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)

        # 8. Transfer function
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)

        # 9. Rank-weighted sum
        weighted_cost = jnp.sum(h_normalized * cost, axis=1)
        return jnp.mean(weighted_cost)

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omega_hat_ = params['omega_hat']

    @property
    def omega_hat_matrix(self):
        """Return the learned :math:`\\hat\\Omega` matrix."""
        if self.omega_hat_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.omega_hat_

    @property
    def lambda_hat_matrix(self):
        """Return :math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T`."""
        if self.omega_hat_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.omega_hat_ @ self.omega_hat_.T

    def predict(self, X):
        """Predict using exponential kernel distance in tangent space.

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

        tangent_flat = self._compute_tangent_vectors(X_m, W_m)
        lambda_hat = jnp.dot(self.omega_hat_, self.omega_hat_.T)
        Lv = jnp.einsum('npd,de->npe', tangent_flat, lambda_hat)
        vLv = jnp.sum(tangent_flat * Lv, axis=2)
        vLv = jnp.clip(vLv, None, 20.0)
        distances = jnp.maximum(jnp.exp(vLv) - 1.0, 0.0)

        return wtac(distances, self.prototype_labels_)

    def _get_quantizable_attrs(self):
        attrs = {'prototypes_': self.prototypes_}
        if self.omega_hat_ is not None:
            attrs['omega_hat_'] = self.omega_hat_
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omega_hat_ is not None:
            arrays['omega_hat_'] = np.asarray(self.omega_hat_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omega_hat_' in arrays:
            self.omega_hat_ = jnp.asarray(arrays['omega_hat_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['omega_hat_scale'] = self.omega_hat_scale
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
