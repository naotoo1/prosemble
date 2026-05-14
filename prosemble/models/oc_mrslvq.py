"""
One-Class Matrix RSLVQ (OC-MRSLVQ) and One-Class Local Matrix RSLVQ (OC-LMRSLVQ).

Extends OC-GLVQ with:
- Omega metric adaptation (global or per-prototype)
- Probabilistic soft-weighting of all prototypes via Gaussian mixture

Instead of using only the nearest prototype (hard argmin like OC-GMLVQ),
all prototypes contribute to the loss via Gaussian proximity weights:

    p(k|x) = exp(-d_k / 2σ²) / Σ_j exp(-d_j / 2σ²)
    μ_k = s · (d_k - θ_k) / (d_k + θ_k)
    loss = mean(Σ_k p(k|x) · sigmoid(μ_k + margin, β))

References
----------
.. [1] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization. Neural
       Computation.
.. [2] Seo, S., & Obermayer, K. (2007). Soft Nearest Prototype
       Classification. IEEE Trans. Neural Networks.
.. [3] Staps et al. (2022). Prototype-based One-Class-Classification
       Learning Using Local Representations. IEEE WSOM+ 2022.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.oc_glvq import OCGLVQ
from prosemble.core.initializers import identity_omega_init
from prosemble.core.activations import sigmoid_beta


class OCMRSLVQ(OCGLVQ):
    """One-Class Matrix Robust Soft LVQ.

    Combines one-class threshold detection with a learned global Omega
    projection matrix and probabilistic soft-weighting of all prototypes.

    All prototypes contribute to the one-class decision via Gaussian
    proximity weights, with distances computed in the Omega-projected space.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture for prototype weighting.
    latent_dim : int, optional
        Dimensionality of the projected space. Default: n_features.
    n_prototypes : int
        Number of prototypes for the target class. Default: 3.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.

    Attributes
    ----------
    omega_ : array of shape (n_features, latent_dim)
        Learned projection matrix.
    thetas_ : array of shape (n_prototypes,)
        Learned per-prototype visibility thresholds.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, sigma=1.0, latent_dim=None, n_prototypes=3,
                 target_label=None, beta=10.0, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            n_prototypes=n_prototypes, target_label=target_label,
            beta=beta, max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, distance_fn=distance_fn,
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
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.omega_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['omega'] = self.omega_
        return params

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)
        n_features = X.shape[1]
        latent_dim = self.latent_dim if self.latent_dim is not None else n_features
        params['omega'] = identity_omega_init(n_features, latent_dim)

        # Reinitialize thetas using omega-projected distances
        target_mask = (y == self._target_label)
        X_target = X[target_mask]
        prototypes = params['prototypes']
        omega = params['omega']
        diff = X_target[:, None, :] - prototypes[None, :, :]
        projected = jnp.einsum('nkd,dl->nkl', diff, omega)
        dists = jnp.sum(projected ** 2, axis=2)
        params['thetas'] = jnp.sqrt(jnp.mean(dists, axis=0) + 1e-10)

        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
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
        thetas = params['thetas']
        omega = params['omega']

        # Omega-projected squared distances: (n, K)
        diff = X[:, None, :] - prototypes[None, :, :]
        projected = jnp.einsum('nkd,dl->nkl', diff, omega)
        distances = jnp.sum(projected ** 2, axis=2)

        # Gaussian weights: p(k|x) for all prototypes
        log_probs = -distances / (2.0 * self.sigma ** 2)
        log_norm = jnp.max(log_probs, axis=1, keepdims=True)
        weights = jnp.exp(log_probs - log_norm)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)

        # Per-prototype OC mu
        s = jnp.where(y == self._target_label, 1.0, -1.0)
        mu = s[:, None] * (distances - thetas[None, :]) / (
            distances + thetas[None, :] + 1e-10
        )

        # Weighted sigmoid loss
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)
        return jnp.mean(jnp.sum(weights * cost, axis=1))

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omega_ = params['omega']

    def decision_function(self, X):
        """Compute target-likeness scores using soft-weighted Omega distances.

        Scores near 1.0 indicate target class, near 0.0 indicate outlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('nkd,dl->nkl', diff, self.omega_)
        distances = jnp.sum(projected ** 2, axis=2)

        # Gaussian weights
        log_probs = -distances / (2.0 * self.sigma ** 2)
        log_norm = jnp.max(log_probs, axis=1, keepdims=True)
        weights = jnp.exp(log_probs - log_norm)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)

        # Per-prototype mu (from target perspective, no sign flip)
        mu = (distances - self.thetas_[None, :]) / (
            distances + self.thetas_[None, :] + 1e-10
        )
        # Weighted score
        weighted_mu = jnp.sum(weights * mu, axis=1)
        return 1.0 - jax.nn.sigmoid(self.beta * weighted_mu)

    @property
    def omega_matrix(self):
        """Return the learned projection matrix Omega."""
        if self.omega_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.omega_

    @property
    def lambda_matrix(self):
        """Return the implicit metric Lambda = Omega^T Omega."""
        return self.omega_matrix.T @ self.omega_matrix

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omega_ is not None:
            attrs.append('omega_')
        return attrs

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
        hp['sigma'] = self.sigma
        hp['latent_dim'] = self.latent_dim
        return hp


class OCLMRSLVQ(OCGLVQ):
    """One-Class Localized Matrix Robust Soft LVQ.

    Each prototype k has its own Omega_k matrix for local metric
    adaptation, combined with probabilistic soft-weighting and
    one-class threshold detection.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture for prototype weighting.
    latent_dim : int, optional
        Latent space dimensionality per prototype. Default: n_features.
    n_prototypes : int
        Number of prototypes for the target class. Default: 3.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.

    Attributes
    ----------
    omegas_ : array of shape (n_prototypes, n_features, latent_dim)
        Learned per-prototype projection matrices.
    thetas_ : array of shape (n_prototypes,)
        Learned per-prototype visibility thresholds.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, sigma=1.0, latent_dim=None, n_prototypes=3,
                 target_label=None, beta=10.0, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            n_prototypes=n_prototypes, target_label=target_label,
            beta=beta, max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, distance_fn=distance_fn,
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
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.omegas_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['omegas'] = self.omegas_
        return params

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)
        n_features = X.shape[1]
        latent_dim = self.latent_dim if self.latent_dim is not None else n_features
        n_protos = params['prototypes'].shape[0]
        omega_single = identity_omega_init(n_features, latent_dim)
        params['omegas'] = jnp.tile(omega_single[None, :, :], (n_protos, 1, 1))

        # Reinitialize thetas using local-omega-projected distances
        target_mask = (y == self._target_label)
        X_target = X[target_mask]
        prototypes = params['prototypes']
        omegas = params['omegas']
        diff = X_target[:, None, :] - prototypes[None, :, :]
        projected = jnp.einsum('nkd,kdl->nkl', diff, omegas)
        dists = jnp.sum(projected ** 2, axis=2)
        params['thetas'] = jnp.sqrt(jnp.mean(dists, axis=0) + 1e-10)

        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
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
        thetas = params['thetas']
        omegas = params['omegas']  # (K, d, l)

        # Local omega-projected squared distances: (n, K)
        diff = X[:, None, :] - prototypes[None, :, :]
        projected = jnp.einsum('nkd,kdl->nkl', diff, omegas)
        distances = jnp.sum(projected ** 2, axis=2)

        # Gaussian weights
        log_probs = -distances / (2.0 * self.sigma ** 2)
        log_norm = jnp.max(log_probs, axis=1, keepdims=True)
        weights = jnp.exp(log_probs - log_norm)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)

        # Per-prototype OC mu
        s = jnp.where(y == self._target_label, 1.0, -1.0)
        mu = s[:, None] * (distances - thetas[None, :]) / (
            distances + thetas[None, :] + 1e-10
        )

        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)
        return jnp.mean(jnp.sum(weights * cost, axis=1))

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omegas_ = params['omegas']

    def decision_function(self, X):
        """Compute target-likeness scores using local Omega distances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('nkd,kdl->nkl', diff, self.omegas_)
        distances = jnp.sum(projected ** 2, axis=2)

        # Gaussian weights
        log_probs = -distances / (2.0 * self.sigma ** 2)
        log_norm = jnp.max(log_probs, axis=1, keepdims=True)
        weights = jnp.exp(log_probs - log_norm)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)

        mu = (distances - self.thetas_[None, :]) / (
            distances + self.thetas_[None, :] + 1e-10
        )
        weighted_mu = jnp.sum(weights * mu, axis=1)
        return 1.0 - jax.nn.sigmoid(self.beta * weighted_mu)

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omegas_ is not None:
            attrs.append('omegas_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omegas_ is not None:
            arrays['omegas_'] = np.asarray(self.omegas_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omegas_' in arrays:
            self.omegas_ = jnp.asarray(arrays['omegas_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma'] = self.sigma
        hp['latent_dim'] = self.latent_dim
        return hp
