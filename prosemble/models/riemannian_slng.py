"""
Supervised Riemannian Localized Matrix Neural Gas (RiemannianSLNG).

Extends RiemannianSRNG with per-prototype metric adaptation in the
tangent space. Each prototype has its own :math:`\\Omega_k` matrix
applied to tangent vectors at that prototype.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.riemannian_smng import RiemannianSMNG
from prosemble.models.prototype_base import SupervisedState
from prosemble.core.activations import sigmoid_beta
from prosemble.core.initializers import identity_omega_init


class RiemannianSLNG(RiemannianSMNG):
    """Supervised Riemannian Localized Matrix Neural Gas.

    Extends RiemannianSRNG with per-prototype metric adaptation. Each
    prototype :math:`w_k` has its own matrix :math:`\\Omega_k` applied
    in the tangent space:

    .. math::

        d(x, w_k) = \\|\\Omega_k \\cdot \\text{Log}_{w_k}(x)_{\\text{flat}}\\|^2

    Since each :math:`\\Omega_k` operates on tangent vectors at :math:`w_k`
    (all in the same tangent space :math:`T_{w_k}M`), this is geometrically
    well-defined.

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance.
    latent_dim : int, optional
        Projection dimensionality for each omega. Default: n_features.
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
            manifold=manifold, latent_dim=latent_dim, beta=beta,
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
        self.omegas_ = None

    def _get_resume_params(self, params):
        gamma = params.get('gamma', jnp.array(self.gamma_final))
        omegas = params.get('omegas', self.omegas_) if self.omegas_ is not None else params.get('omegas')
        return {
            'prototypes': params['prototypes'],
            'omegas': omegas,
            'gamma': gamma,
        }

    def _init_state(self, X, y, key):
        # Call RiemannianSRNG._init_state (skip RiemannianSMNG)
        state, params, proto_labels = RiemannianSMNG.__bases__[0]._init_state(self, X, y, key)

        d_flat = X.shape[1]
        n_protos = params['prototypes'].shape[0]
        latent_dim = self.latent_dim if self.latent_dim is not None else d_flat
        omega_one = identity_omega_init(d_flat, latent_dim)
        omegas = jnp.tile(omega_one[None, :, :], (n_protos, 1, 1))

        params = {**params, 'omegas': omegas}
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
        omegas = params['omegas']  # (p, d_flat, l)
        gamma = params['gamma']

        n = X.shape[0]
        p = prototypes.shape[0]
        X_m = self._reshape_to_manifold(X, n)
        W_m = self._reshape_to_manifold(prototypes, p)

        # 1. Per-prototype omega distance in tangent space
        tangent_flat = self._compute_tangent_vectors(X_m, W_m)  # (n, p, d_flat)
        projected = jnp.einsum('npd,pdl->npl', tangent_flat, omegas)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)

        # 2-8: NG+GLVQ (identical pattern)
        same_class = (y[:, None] == proto_labels[None, :])
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)

        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)

        h = jnp.exp(-ranks / (gamma + 1e-10))
        h = jnp.where(same_class, h, 0.0)

        C = jnp.sum(h, axis=1, keepdims=True)
        h_normalized = h / (C + 1e-10)

        d_diff = jnp.where(~same_class, distances, INF)
        dm = jnp.min(d_diff, axis=1)

        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)

        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)

        weighted_cost = jnp.sum(h_normalized * cost, axis=1)
        return jnp.mean(weighted_cost)

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        # Call RiemannianSRNG._extract_results (skip RiemannianSMNG)
        RiemannianSMNG.__bases__[0]._extract_results(
            self, params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omegas_ = params['omegas']

    def predict(self, X):
        """Predict using per-prototype tangent-space omega metric.

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
        projected = jnp.einsum('npd,pdl->npl', tangent_flat, self.omegas_)
        distances = jnp.sum(projected ** 2, axis=2)

        from prosemble.core.competitions import wtac
        return wtac(distances, self.prototype_labels_)

    def _get_quantizable_attrs(self):
        return {'prototypes_': self.prototypes_, 'omegas_': self.omegas_}

    def _get_fitted_arrays(self):
        arrays = RiemannianSMNG.__bases__[0]._get_fitted_arrays(self)
        if self.omegas_ is not None:
            arrays['omegas_'] = np.asarray(self.omegas_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        RiemannianSMNG.__bases__[0]._set_fitted_arrays(self, arrays)
        if 'omegas_' in arrays:
            self.omegas_ = jnp.asarray(arrays['omegas_'])
