"""
Riemannian Differentiating Kernel Matrix STNG (RiemannianDKMSTNG).

Combines RiemannianSTNG (tangent subspace projection) with an exponential
kernel using a learned transformation matrix on the subspace residual:

.. math::

    d_\\kappa^2(x, w_k) = \\exp\\left(
        r^T \\hat\\Lambda r
    \\right) - 1

where :math:`r = (I - \\Omega_k \\Omega_k^T) \\cdot v` is the subspace
residual, :math:`v = \\text{Log}_{w_k}(x)_{\\text{flat}}`, and
:math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T` is a learned PSD matrix.

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.riemannian_stng import RiemannianSTNG
from prosemble.models.prototype_base import SupervisedState
from prosemble.core.activations import sigmoid_beta
from prosemble.core.initializers import identity_omega_init
from prosemble.core.competitions import wtac


class RiemannianDKMSTNG(RiemannianSTNG):
    """Riemannian Differentiating Kernel Matrix STNG.

    Extends RiemannianSTNG with an exponential kernel on the tangent
    subspace residual. Each prototype has an orthonormal subspace basis
    :math:`\\Omega_k`, and a shared :math:`\\hat\\Lambda = \\hat\\Omega
    \\hat\\Omega^T` provides metric adaptation on the residual:

    .. math::

        d_\\kappa^2(x, w_k) = \\exp\\left(
            r^T \\hat\\Lambda r
        \\right) - 1

    where :math:`r = (I - \\Omega_k \\Omega_k^T) \\cdot v`.

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance.
    kernel_latent_dim : int, optional
        Dimensionality for the kernel's omega_hat. Default: d_flat.
    omega_hat_scale : float
        Scale for omega_hat initialization. Default: 0.1.
    subspace_dim : int, optional
        Tangent subspace dimensionality. Default: d_flat - 1.
    beta : float
        Transfer function steepness.
    gamma_init : float, optional
        Initial neighborhood range.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step decay factor.
    tau : float
        Injectivity radius safety factor. Default: 0.95.
    n_prototypes_per_class : int
        Number of prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    epsilon : float
        Convergence threshold.
    random_seed : int
        Random seed.
    optimizer : str or optax optimizer, optional
        Default: 'adam'.
    transfer_fn : callable, optional
        Transfer function.
    margin : float
        Margin for loss.
    callbacks : list, optional
        Callback objects.
    use_scan : bool
        Default: False.
    batch_size : int, optional
        Mini-batch size.
    lr_scheduler : str or optax.Schedule, optional
        Learning rate schedule.
    lr_scheduler_kwargs : dict, optional
        LR scheduler kwargs.
    prototypes_initializer : str or callable, optional
        Prototype initialization.
    patience : int, optional
        Early stopping patience.
    restore_best : bool
        Restore best parameters. Default: False.
    class_weight : dict or 'balanced', optional
        Class weights.
    gradient_accumulation_steps : int, optional
        Gradient accumulation.
    ema_decay : float, optional
        EMA decay.
    freeze_params : list of str, optional
        Frozen parameters.
    lookahead : dict, optional
        Lookahead config.
    mixed_precision : str or None, optional
        Mixed precision dtype.

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.

    See Also
    --------
    RiemannianSTNG : Base Riemannian tangent Neural Gas.
    RiemannianDKSTNG : Gaussian kernel variant (no matrix kernel).
    RiemannianDKRSTNG : Relevance kernel variant.
    """

    def __init__(self, manifold, kernel_latent_dim=None, omega_hat_scale=0.1,
                 subspace_dim=None, beta=10.0, gamma_init=None,
                 gamma_final=0.01, gamma_decay=None, tau=0.95,
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, optimizer='adam',
                 transfer_fn=None, margin=0.0, callbacks=None,
                 use_scan=False, batch_size=None, lr_scheduler=None,
                 lr_scheduler_kwargs=None, prototypes_initializer=None,
                 patience=None, restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            manifold=manifold, subspace_dim=subspace_dim, beta=beta,
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
        self.kernel_latent_dim = kernel_latent_dim
        self.omega_hat_scale = omega_hat_scale
        self.omega_hat_ = None

    def _get_resume_params(self, params):
        base = super()._get_resume_params(params)
        base['omega_hat'] = self.omega_hat_
        return base

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)

        # omega_hat operates on the residual (d_flat dimensional)
        d_flat = X.shape[1]
        kld = self.kernel_latent_dim if self.kernel_latent_dim is not None else d_flat
        omega_hat = self.omega_hat_scale * identity_omega_init(d_flat, kld)

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
        omegas = params['omegas']
        gamma = params['gamma']
        omega_hat = params['omega_hat']
        lambda_hat = jnp.dot(omega_hat, omega_hat.T)  # (d_flat, d_flat)

        n = X.shape[0]
        p = prototypes.shape[0]
        X_m = self._reshape_to_manifold(X, n)
        W_m = self._reshape_to_manifold(prototypes, p)

        # 1. Tangent subspace residual
        tangent_flat = self._compute_tangent_vectors(X_m, W_m)
        proj = jnp.einsum('npd,pds->nps', tangent_flat, omegas)
        recon = jnp.einsum('nps,pds->npd', proj, omegas)
        residual = tangent_flat - recon  # (n, p, d_flat)

        # 2. Exponential kernel: exp(r^T Lambda_hat r) - 1
        Lr = jnp.einsum('npd,de->npe', residual, lambda_hat)  # (n, p, d_flat)
        rLr = jnp.sum(residual * Lr, axis=2)  # (n, p)
        rLr = jnp.clip(rLr, None, 20.0)  # prevent exp overflow
        distances = jnp.maximum(jnp.exp(rLr) - 1.0, 0.0)

        # 3. NG ranking + GLVQ loss
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
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omega_hat_ = params['omega_hat']

    @property
    def omega_hat_matrix(self):
        """Return the learned kernel :math:`\\hat\\Omega` matrix."""
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
        """Predict using exponential kernel on subspace residual.

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
        proj = jnp.einsum('npd,pds->nps', tangent_flat, self.omegas_)
        recon = jnp.einsum('nps,pds->npd', proj, self.omegas_)
        residual = tangent_flat - recon

        lambda_hat = jnp.dot(self.omega_hat_, self.omega_hat_.T)
        Lr = jnp.einsum('npd,de->npe', residual, lambda_hat)
        rLr = jnp.sum(residual * Lr, axis=2)
        rLr = jnp.clip(rLr, None, 20.0)
        distances = jnp.maximum(jnp.exp(rLr) - 1.0, 0.0)

        return wtac(distances, self.prototype_labels_)

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if isinstance(attrs, dict):
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
        if self.kernel_latent_dim is not None:
            hp['kernel_latent_dim'] = self.kernel_latent_dim
        return hp
