"""
Riemannian Differentiating Kernel SMNG (RiemannianDKSMNG).

Combines RiemannianSMNG (global omega metric in tangent space) with a
Gaussian kernel wrapping for adaptive per-prototype bandwidth:

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{\\|\\Omega \\cdot \\text{Log}_{w_k}(x)_{\\text{flat}}\\|^2}
              {2\\sigma_k^2}
    \\right)\\right)

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.riemannian_smng import RiemannianSMNG
from prosemble.models.prototype_base import SupervisedState
from prosemble.core.activations import sigmoid_beta
from prosemble.core.initializers import identity_omega_init
from prosemble.core.competitions import wtac


class RiemannianDKSMNG(RiemannianSMNG):
    """Riemannian Differentiating Kernel SMNG.

    Extends RiemannianSMNG with a Gaussian kernel wrapping the
    omega-projected tangent distance. Each prototype has a learnable
    bandwidth :math:`\\sigma_k`:

    .. math::

        d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
            -\\frac{\\|\\Omega \\cdot v\\|^2}{2\\sigma_k^2}
        \\right)\\right)

    where :math:`v = \\text{Log}_{w_k}(x)_{\\text{flat}}`.

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance.
    sigma_init : str or float
        Initialization strategy for per-prototype bandwidths.
        'median' (default): per-class median of omega-projected distances.
        'mean': per-class mean.
        float: fixed value for all prototypes.
    sigma_min : float
        Lower bound for sigma. Default: 1e-3.
    latent_dim : int, optional
        Projection dimensionality for omega. Default: d_flat.
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
    RiemannianSMNG : Base Riemannian matrix Neural Gas.
    RiemannianDKGLVQ : Gaussian kernel on geodesic distance (no omega).
    """

    def __init__(self, manifold, sigma_init='median', sigma_min=1e-3,
                 latent_dim=None, beta=10.0, gamma_init=None,
                 gamma_final=0.01, gamma_decay=None, lr_ratio=0.5,
                 tau=0.95,
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
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.lr_ratio = lr_ratio
        self.sigmas_ = None

    def _estimate_sigmas(self, X, y, params, proto_labels):
        """Estimate per-prototype bandwidths from omega-projected distances."""
        if isinstance(self.sigma_init, (int, float)):
            return jnp.full(params['prototypes'].shape[0], float(self.sigma_init))

        prototypes = params['prototypes']
        omega = params['omega']
        n = X.shape[0]
        p = prototypes.shape[0]
        X_m = self._reshape_to_manifold(X, n)
        W_m = self._reshape_to_manifold(prototypes, p)

        tangent_flat = self._compute_tangent_vectors(X_m, W_m)
        projected = jnp.einsum('npd,dl->npl', tangent_flat, omega)
        raw_distances = jnp.sum(projected ** 2, axis=2)  # (n, p)

        sigmas = []
        for k in range(p):
            label_k = proto_labels[k]
            class_mask = (y == label_k)
            dists_k = jnp.sqrt(jnp.maximum(raw_distances[class_mask, k], 0.0))
            if self.sigma_init == 'median':
                sigma_k = jnp.median(dists_k)
            else:
                sigma_k = jnp.mean(dists_k)
            sigmas.append(jnp.maximum(sigma_k, self.sigma_min))
        return jnp.array(sigmas)

    def _get_resume_params(self, params):
        base = super()._get_resume_params(params)
        base['sigmas'] = self.sigmas_
        return base

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)

        sigmas = self._estimate_sigmas(X, y, params, proto_labels)
        params = {**params, 'sigmas': sigmas}

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
        sigmas = jnp.maximum(params['sigmas'], self.sigma_min)

        n = X.shape[0]
        p = prototypes.shape[0]
        X_m = self._reshape_to_manifold(X, n)
        W_m = self._reshape_to_manifold(prototypes, p)

        # 1. Omega-projected tangent distance
        tangent_flat = self._compute_tangent_vectors(X_m, W_m)
        projected = jnp.einsum('npd,dl->npl', tangent_flat, omega)
        raw_distances = jnp.sum(projected ** 2, axis=2)  # (n, p)

        # 2. Apply Gaussian kernel
        K = jnp.exp(-raw_distances / (2.0 * sigmas[None, :] ** 2))
        distances = 2.0 * (1.0 - K)

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

        # Separate learning rates (Hammer et al. 2003: epsilon^- = lr_ratio * epsilon^+)
        # Scale gradient through dm by lr_ratio; forward pass unchanged.
        dm = jax.lax.stop_gradient(dm) + self.lr_ratio * (
            dm - jax.lax.stop_gradient(dm))

        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)

        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)

        weighted_cost = jnp.sum(h_normalized * cost, axis=1)
        return jnp.mean(weighted_cost)

    def _post_update(self, params):
        params = super()._post_update(params)
        params['sigmas'] = jnp.maximum(params['sigmas'], self.sigma_min)
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.sigmas_ = params['sigmas']

    @property
    def kernel_bandwidths(self):
        """Return the learned per-prototype bandwidths."""
        if self.sigmas_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.sigmas_

    def predict(self, X):
        """Predict using kernel-wrapped omega-projected tangent distance.

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
        raw_distances = jnp.sum(projected ** 2, axis=2)

        sigmas = jnp.maximum(self.sigmas_, self.sigma_min)
        K = jnp.exp(-raw_distances / (2.0 * sigmas[None, :] ** 2))
        distances = 2.0 * (1.0 - K)

        return wtac(distances, self.prototype_labels_)

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if isinstance(attrs, dict):
            if self.sigmas_ is not None:
                attrs['sigmas_'] = self.sigmas_
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.sigmas_ is not None:
            arrays['sigmas_'] = np.asarray(self.sigmas_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'sigmas_' in arrays:
            self.sigmas_ = jnp.asarray(arrays['sigmas_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma_init'] = self.sigma_init
        hp['sigma_min'] = self.sigma_min
        hp['lr_ratio'] = self.lr_ratio
        return hp
