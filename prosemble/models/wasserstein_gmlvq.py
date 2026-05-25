"""
Wasserstein GMLVQ (WGMLVQ).

Combines Wasserstein distributional prototypes with a global linear
transformation :math:`\\Omega`. The squared 2-Wasserstein distance in
the projected space is:

.. math::

    W_2^2(x, k) = \\|\\Omega(x - \\mu_k)\\|^2 + \\sum_j \\sigma_{kj}^2

The metric adaptation matrix :math:`\\Lambda = \\Omega^T \\Omega` learns
discriminative feature combinations, while the variance captures
per-prototype uncertainty.

References
----------
.. [1] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization. Neural
       Computation.
.. [2] Villani, C. (2009). Optimal Transport: Old and New. Springer.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel, SupervisedState
from prosemble.core.losses import glvq_loss_with_transfer
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init
from prosemble.core.distance import wasserstein2_omega_distance_matrix


class WGMLVQ(SupervisedPrototypeModel):
    """Wasserstein Generalized Matrix Learning Vector Quantization.

    Extends WGLVQ with a learned global projection matrix :math:`\\Omega`
    that maps data into a discriminative subspace before computing the
    Euclidean component of the Wasserstein distance.

    Parameters
    ----------
    latent_dim : int, optional
        Dimension of the projected space. Default: input dimension.
    beta : float
        Transfer function steepness. Default: 10.0.
    n_prototypes_per_class : int or dict
        Number of prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    epsilon : float
        Convergence threshold.
    random_seed : int
        Random seed.
    optimizer : str or optax optimizer
        Optimizer. Default: 'adam'.
    transfer_fn : callable, optional
        Transfer function.
    margin : float
        Margin for loss.
    callbacks : list, optional
        Training callbacks.
    use_scan : bool
        Use ``jax.lax.scan``.
    batch_size : int, optional
        Mini-batch size.
    lr_scheduler : str or optax.Schedule, optional
        Learning rate schedule.
    lr_scheduler_kwargs : dict, optional
        Scheduler kwargs.
    prototypes_initializer : str or callable, optional
        Prototype initialization.
    patience : int, optional
        Early stopping patience.
    restore_best : bool
        Restore best parameters.
    class_weight : dict or 'balanced', optional
        Class weights.
    gradient_accumulation_steps : int, optional
        Gradient accumulation.
    ema_decay : float, optional
        EMA decay.
    freeze_params : list of str, optional
        Frozen parameter groups.
    lookahead : dict, optional
        Lookahead settings.
    mixed_precision : str, optional
        Mixed precision dtype.

    Attributes
    ----------
    prototype_means_ : array of shape (n_prototypes, n_features)
        Learned means.
    prototype_variances_ : array of shape (n_prototypes, n_features)
        Learned variances.
    omega_ : array of shape (n_features, latent_dim)
        Learned projection matrix.
    """

    def __init__(self, latent_dim=None, beta=10.0,
                 n_prototypes_per_class=1,
                 max_iter=100, lr=0.01, epsilon=1e-6, random_seed=42,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
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
        self.latent_dim = latent_dim
        self.beta = beta
        self.omega_ = None
        self.prototype_means_ = None
        self.prototype_variances_ = None

    def _get_resume_params(self, params):
        return {
            'prototypes': params['prototypes'],
            'log_variances': self._log_variances_,
            'omega': self.omega_,
        }

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )

        omega = identity_omega_init(n_features, latent_dim)

        # Initialize log-variances from per-class variance
        n_protos = prototypes.shape[0]
        class_vars = []
        for i in range(n_protos):
            mask = y == proto_labels[i]
            class_data = X[mask]
            var = jnp.var(class_data, axis=0)
            class_vars.append(var)
        class_vars = jnp.stack(class_vars)
        log_variances = jnp.log(jnp.maximum(class_vars, 1e-6))

        params = {
            'prototypes': prototypes,
            'log_variances': log_variances,
            'omega': omega,
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
        means = params['prototypes']
        log_variances = params['log_variances']
        omega = params['omega']
        distances = wasserstein2_omega_distance_matrix(
            X, means, log_variances, omega
        )
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _compute_distances_for_rejection(self, X):
        """Omega-projected W2 distances for reject option."""
        log_vars = jnp.log(jnp.maximum(self.prototype_variances_, 1e-6))
        return wasserstein2_omega_distance_matrix(
            X, self.prototypes_, log_vars, self.omega_
        )

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omega_ = params['omega']
        self.prototype_means_ = params['prototypes']
        self.prototype_variances_ = jnp.exp(params['log_variances'])
        self._log_variances_ = params['log_variances']

    @property
    def omega_matrix(self):
        """Return the learned :math:`\\Omega` matrix."""
        if self.omega_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_

    @property
    def lambda_matrix(self):
        """Return :math:`\\Lambda = \\Omega^T \\Omega` (relevance matrix)."""
        if self.omega_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_.T @ self.omega_

    def predict(self, X):
        """Predict class labels using omega-projected W2 distance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        log_vars = jnp.log(jnp.maximum(self.prototype_variances_, 1e-6))
        distances = wasserstein2_omega_distance_matrix(
            X, self.prototypes_, log_vars, self.omega_
        )
        return wtac(distances, self.prototype_labels_)

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omega_ is not None:
            attrs.append('omega_')
        if self.prototype_variances_ is not None:
            attrs.append('prototype_variances_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omega_ is not None:
            arrays['omega_'] = np.asarray(self.omega_)
        if self.prototype_variances_ is not None:
            arrays['prototype_variances_'] = np.asarray(self.prototype_variances_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omega_' in arrays:
            self.omega_ = jnp.asarray(arrays['omega_'])
        if 'prototype_variances_' in arrays:
            self.prototype_variances_ = jnp.asarray(arrays['prototype_variances_'])
            self._log_variances_ = jnp.log(jnp.maximum(self.prototype_variances_, 1e-6))
            self.prototype_means_ = self.prototypes_

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
