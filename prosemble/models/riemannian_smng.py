"""
Supervised Riemannian Matrix Neural Gas (RiemannianSMNG).

Extends RiemannianSRNG with a global learned metric in the tangent space.
For each prototype, the tangent vector from prototype to data point is
computed via the manifold's logarithmic map, then projected through a
shared matrix :math:`\\Omega` before computing distances.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.riemannian_srng import RiemannianSRNG
from prosemble.models.prototype_base import SupervisedState
from prosemble.core.activations import sigmoid_beta
from prosemble.core.initializers import identity_omega_init


class RiemannianSMNG(RiemannianSRNG):
    """Supervised Riemannian Matrix Neural Gas.

    Extends RiemannianSRNG with a global metric adaptation matrix
    :math:`\\Omega` applied in the tangent space. The distance is:

    .. math::

        d(x, w_k) = \\|\\Omega \\cdot \\text{Log}_{w_k}(x)_{\\text{flat}}\\|^2

    where :math:`\\text{Log}_{w_k}(x)` is the logarithmic map at prototype
    :math:`w_k`, flattened to a vector.

    The learned relevance matrix :math:`\\Lambda = \\Omega^T \\Omega`
    captures feature correlations in the tangent space.

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance.
    latent_dim : int, optional
        Projection dimensionality for omega. Default: n_features (square).
    beta : float
        Transfer function steepness.
    gamma_init : float, optional
        Initial neighborhood range. Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step decay factor for gamma.
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
    """

    def __init__(self, manifold, latent_dim=None, beta=10.0,
                 gamma_init=None, gamma_final=0.01, gamma_decay=None,
                 tau=0.95, n_prototypes_per_class=1, max_iter=100,
                 lr=0.01, epsilon=1e-6, random_seed=42,
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
            gamma_decay=gamma_decay, tau=tau,
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
        self.omega_ = None

    def _diff_log_map(self, w, x):
        """Differentiable log map for a single (base, target) pair.

        Dispatches to the appropriate differentiable log map based on
        manifold type.
        """
        from prosemble.core.manifolds import SO, SPD, Grassmannian
        from prosemble.models.riemannian_srng import (
            _so_log_map_diff, _spd_log_map_diff, _grassmannian_log_map_diff,
        )
        if isinstance(self.manifold, SO):
            return _so_log_map_diff(w, x)
        elif isinstance(self.manifold, SPD):
            return _spd_log_map_diff(w, x)
        elif isinstance(self.manifold, Grassmannian):
            return _grassmannian_log_map_diff(w, x)
        else:
            return self.manifold.log_map(w, x)

    def _compute_tangent_vectors(self, X_manifold, W_manifold):
        """Compute tangent vectors from prototypes to data via log map.

        Uses differentiable log map implementations that support autodiff.

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
        # log_matrix(W, X) → (p, n, *point_shape)
        tangents = log_matrix(W_manifold, X_manifold)
        # Transpose to (n, p, *point_shape)
        tangents = jnp.moveaxis(tangents, 0, 1)
        # Flatten to (n, p, d_flat)
        n = X_manifold.shape[0]
        p = W_manifold.shape[0]
        return tangents.reshape(n, p, -1)

    def _get_resume_params(self, params):
        gamma = params.get('gamma', jnp.array(self.gamma_final))
        omega = params.get('omega', self.omega_) if self.omega_ is not None else params.get('omega')
        return {
            'prototypes': params['prototypes'],
            'omega': omega,
            'gamma': gamma,
        }

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)

        d_flat = X.shape[1]
        latent_dim = self.latent_dim if self.latent_dim is not None else d_flat
        omega = identity_omega_init(d_flat, latent_dim)

        params = {**params, 'omega': omega}
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
        omega = params['omega']
        gamma = params['gamma']

        n = X.shape[0]
        p = prototypes.shape[0]
        X_m = self._reshape_to_manifold(X, n)
        W_m = self._reshape_to_manifold(prototypes, p)

        # 1. Tangent vectors via log map, then global omega projection
        tangent_flat = self._compute_tangent_vectors(X_m, W_m)  # (n, p, d_flat)
        projected = jnp.einsum('npd,dl->npl', tangent_flat, omega)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)

        # 2. Compute ranks within same-class prototypes
        same_class = (y[:, None] == proto_labels[None, :])
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)

        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)

        # 3. Neighborhood function
        h = jnp.exp(-ranks / (gamma + 1e-10))
        h = jnp.where(same_class, h, 0.0)

        # 4. Normalize
        C = jnp.sum(h, axis=1, keepdims=True)
        h_normalized = h / (C + 1e-10)

        # 5. Closest different-class distance
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

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omega_ = params['omega']

    def predict(self, X):
        """Predict using tangent-space omega metric.

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
        projected = jnp.einsum('npd,dl->npl', tangent_flat, self.omega_)
        distances = jnp.sum(projected ** 2, axis=2)

        from prosemble.core.competitions import wtac
        return wtac(distances, self.prototype_labels_)

    def relevance_matrix(self):
        """Return learned relevance matrix Lambda = Omega^T Omega.

        Returns
        -------
        array of shape (d_flat, d_flat)
        """
        return self.omega_.T @ self.omega_

    def _get_quantizable_attrs(self):
        return {'prototypes_': self.prototypes_, 'omega_': self.omega_}

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omega_ is not None:
            arrays['omega_'] = np.asarray(self.omega_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omega_' in arrays:
            self.omega_ = jnp.asarray(arrays['omega_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['latent_dim'] = self.latent_dim
        return hp
