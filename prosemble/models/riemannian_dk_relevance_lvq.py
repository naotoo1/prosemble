"""
Riemannian Differentiating Kernel GRLVQ (RiemannianDKGRLVQ).

Combines Riemannian manifold geometry with relevance-weighted Gaussian
kernel distance in tangent space:

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{\\sum_j \\lambda_j v_j^2}{2\\sigma_k^2}
    \\right)\\right)

where :math:`v = \\text{Log}_{w_k}(x)_{\\text{flat}}` is the tangent vector,
:math:`\\lambda = \\text{softmax}(\\text{relevances})`, and :math:`\\sigma_k`
is a per-prototype bandwidth.

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
from prosemble.core.competitions import wtac
from prosemble.core.manifolds import SO, SPD, Grassmannian, HyperbolicPoincare
from prosemble.models.riemannian_srng import (
    _so_log_map_diff, _spd_log_map_diff, _grassmannian_log_map_diff,
    _hyperbolic_log_map_diff,
)


class RiemannianDKGRLVQ(RiemannianSRNG):
    """Riemannian Differentiating Kernel GRLVQ.

    Extends RiemannianSRNG with relevance-weighted Gaussian kernel
    distance in tangent space. Each prototype :math:`w_k` has a learnable
    bandwidth :math:`\\sigma_k`, and a shared relevance vector
    :math:`\\lambda` weights the tangent space features.

    .. math::

        d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
            -\\frac{\\sum_j \\lambda_j v_j^2}{2\\sigma_k^2}
        \\right)\\right)

    where :math:`v = \\text{Log}_{w_k}(x)_{\\text{flat}}`.

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance defining the geometry.
    sigma_init : str or float
        Initialization strategy for per-prototype bandwidths.
        'median' (default): per-class median geodesic distance.
        'mean': per-class mean geodesic distance.
        float: fixed value for all prototypes.
    sigma_min : float
        Lower bound for sigma to prevent bandwidth collapse. Default: 1e-3.
    beta : float
        Transfer function steepness parameter.
    gamma_init : float, optional
        Initial neighborhood range for NG cooperation.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay factor for gamma.
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
    RiemannianDKGLVQ : Gaussian kernel variant without relevance weighting.
    """

    def __init__(self, manifold, sigma_init='median', sigma_min=1e-3,
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
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.sigmas_ = None
        self.relevances_ = None

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

    def _estimate_sigmas(self, X_flat, y, prototypes_flat, proto_labels):
        """Estimate per-prototype bandwidths from manifold distances."""
        if isinstance(self.sigma_init, (int, float)):
            return jnp.full(prototypes_flat.shape[0], float(self.sigma_init))

        n_protos = prototypes_flat.shape[0]
        n = X_flat.shape[0]
        X_m = self._reshape_to_manifold(X_flat, n)
        W_m = self._reshape_to_manifold(prototypes_flat, n_protos)

        dist_matrix = self._geodesic_distances(X_m, W_m)

        sigmas = []
        for k in range(n_protos):
            label_k = proto_labels[k]
            class_mask = (y == label_k)
            dists_k = jnp.sqrt(jnp.maximum(dist_matrix[class_mask, k], 0.0))
            if self.sigma_init == 'median':
                sigma_k = jnp.median(dists_k)
            else:
                sigma_k = jnp.mean(dists_k)
            sigmas.append(jnp.maximum(sigma_k, self.sigma_min))
        return jnp.array(sigmas)

    def _get_resume_params(self, params):
        base = super()._get_resume_params(params)
        base['sigmas'] = self.sigmas_
        base['relevances'] = self.relevances_
        return base

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)

        d_flat = X.shape[1]
        sigmas = self._estimate_sigmas(X, y, params['prototypes'], proto_labels)
        relevances = jnp.zeros(d_flat)  # uniform under softmax

        params = {**params, 'sigmas': sigmas, 'relevances': relevances}

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
        sigmas = jnp.maximum(params['sigmas'], self.sigma_min)
        lam = jax.nn.softmax(params['relevances'])

        n = X.shape[0]
        p = prototypes.shape[0]
        X_m = self._reshape_to_manifold(X, n)
        W_m = self._reshape_to_manifold(prototypes, p)

        # 1. Tangent vectors via log map
        tangent_flat = self._compute_tangent_vectors(X_m, W_m)  # (n, p, d_flat)

        # 2. Relevance-weighted squared norms
        weighted_sq = jnp.sum(
            lam[None, None, :] * tangent_flat ** 2, axis=2
        )  # (n, p)

        # 3. Apply Gaussian kernel
        K = jnp.exp(-weighted_sq / (2.0 * sigmas[None, :] ** 2))
        distances = 2.0 * (1.0 - K)

        # 4. Compute ranks within same-class prototypes
        same_class = (y[:, None] == proto_labels[None, :])
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)

        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)

        # 5. Neighborhood function h = exp(-rank / gamma)
        h = jnp.exp(-ranks / (gamma + 1e-10))
        h = jnp.where(same_class, h, 0.0)

        # 6. Normalize per sample
        C = jnp.sum(h, axis=1, keepdims=True)
        h_normalized = h / (C + 1e-10)

        # 7. Closest different-class prototype distance
        d_diff = jnp.where(~same_class, distances, INF)
        dm = jnp.min(d_diff, axis=1)

        # 8. GLVQ mu
        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)

        # 9. Transfer function
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)

        # 10. Rank-weighted sum
        weighted_cost = jnp.sum(h_normalized * cost, axis=1)
        return jnp.mean(weighted_cost)

    def _post_update(self, params):
        params = super()._post_update(params)
        params['sigmas'] = jnp.maximum(params['sigmas'], self.sigma_min)
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.sigmas_ = params['sigmas']
        self.relevances_ = params['relevances']

    @property
    def kernel_bandwidths(self):
        """Return the learned per-prototype bandwidths."""
        if self.sigmas_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.sigmas_

    @property
    def relevance_profile(self):
        """Return the learned relevance weights (normalized via softmax)."""
        if self.relevances_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return jax.nn.softmax(self.relevances_)

    def predict(self, X):
        """Predict using relevance-weighted kernel distance in tangent space.

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
        lam = jax.nn.softmax(self.relevances_)
        weighted_sq = jnp.sum(
            lam[None, None, :] * tangent_flat ** 2, axis=2
        )

        sigmas = jnp.maximum(self.sigmas_, self.sigma_min)
        K = jnp.exp(-weighted_sq / (2.0 * sigmas[None, :] ** 2))
        distances = 2.0 * (1.0 - K)

        return wtac(distances, self.prototype_labels_)

    def _get_quantizable_attrs(self):
        attrs = {'prototypes_': self.prototypes_}
        if self.sigmas_ is not None:
            attrs['sigmas_'] = self.sigmas_
        if self.relevances_ is not None:
            attrs['relevances_'] = self.relevances_
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.sigmas_ is not None:
            arrays['sigmas_'] = np.asarray(self.sigmas_)
        if self.relevances_ is not None:
            arrays['relevances_'] = np.asarray(self.relevances_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'sigmas_' in arrays:
            self.sigmas_ = jnp.asarray(arrays['sigmas_'])
        if 'relevances_' in arrays:
            self.relevances_ = jnp.asarray(arrays['relevances_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma_init'] = self.sigma_init
        hp['sigma_min'] = self.sigma_min
        return hp
