"""
Matrix SVQ-OCC (SVQ-OCC-M).

Extends SVQ-OCC with a global Omega matrix (GMLVQ-style metric adaptation).
The distance becomes:

    d_Ω(x, w_k) = ||Ω(x - w_k)||²

where Ω is a learned (d × l) projection matrix. The implicit metric
Λ = Ω^T Ω captures feature correlations for one-class classification.

References
----------
.. [1] Staps et al. (2022). Prototype-based One-Class-Classification
       Learning Using Local Representations. IEEE WSOM+ 2022.
.. [2] Schneider, Biehl, Hammer (2009). Adaptive Relevance Matrices
       in Learning Vector Quantization. Neural Computation, 21(12).
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.svq_occ import SVQOCC
from prosemble.core.initializers import identity_omega_init


class SVQOCC_M(SVQOCC):
    """Matrix SVQ-OCC with global Omega projection.

    Learns a global linear projection Ω that captures feature
    correlations for one-class classification.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of the projected space. Default: n_features.
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    alpha : float
        Balance between representation and classification. Default: 0.5.
    cost_function : str
        'contrastive', 'brier', or 'cross_entropy'. Default: 'contrastive'.
    response_type : str
        'gaussian', 'student_t', or 'uniform'. Default: 'gaussian'.

    Attributes
    ----------
    omega_ : array of shape (n_features, latent_dim)
        Learned projection matrix.

    See Also
    --------
    SVQOCC : Full SVQ-OCC parameter documentation.
    SupervisedPrototypeModel : Full list of base parameters.
    """

    def __init__(self, latent_dim=None, n_prototypes=3, target_label=None,
                 alpha=0.5, cost_function='contrastive',
                 response_type='gaussian', sigma=0.1, gamma_resp=1.0,
                 nu=1.0, lambda_init=None, lambda_final=0.01,
                 lambda_decay=None, max_iter=100, lr=0.01, epsilon=1e-6,
                 random_seed=42, distance_fn=None, optimizer='adam',
                 transfer_fn=None, margin=0.0, callbacks=None,
                 use_scan=True, batch_size=None, lr_scheduler=None,
                 lr_scheduler_kwargs=None, prototypes_initializer=None,
                 patience=None, restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None, mixed_precision=None):
        super().__init__(
            n_prototypes=n_prototypes, target_label=target_label,
            alpha=alpha, cost_function=cost_function,
            response_type=response_type, sigma=sigma,
            gamma_resp=gamma_resp, nu=nu, lambda_init=lambda_init,
            lambda_final=lambda_final, lambda_decay=lambda_decay,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
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
        # Reinitialize optimizer with omega
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
        lambda_ng = params['lambda_ng']
        omega = params['omega']

        n_protos = prototypes.shape[0]

        # Omega-projected squared distances
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, K, d)
        projected = jnp.einsum('nkd,dl->nkl', diff, omega)  # (n, K, l)
        sq_distances = jnp.sum(projected ** 2, axis=2)  # (n, K)

        target_mask = (y == self._target_label)

        # Representation cost R (target only)
        order = jnp.argsort(sq_distances, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)
        h_ng = jnp.exp(-ranks / (lambda_ng + 1e-10))
        R_per_sample = jnp.sum(h_ng * sq_distances, axis=1)
        R_per_sample = jnp.where(target_mask, R_per_sample, 0.0)
        R = jnp.sum(R_per_sample) / (jnp.sum(target_mask) + 1e-10)

        # Classification cost C
        if self.response_type == 'gaussian':
            p_k = jax.nn.softmax(-self.gamma_resp * sq_distances, axis=1)
        elif self.response_type == 'student_t':
            p_unnorm = (1.0 + sq_distances / self.nu) ** (-(self.nu + 1) / 2)
            p_k = p_unnorm / (jnp.sum(p_unnorm, axis=1, keepdims=True) + 1e-10)
        else:
            p_k = jnp.ones_like(sq_distances) / n_protos

        thetas_pos = jnp.maximum(thetas, 1e-6)
        heaviside = jax.nn.sigmoid(
            (thetas_pos[None, :] - sq_distances) / (self.sigma + 1e-10)
        )
        total_resp = jnp.clip(
            jnp.sum(p_k * heaviside, axis=1), 1e-10, 1.0 - 1e-10
        )

        y_binary = target_mask.astype(jnp.float32)
        if self.cost_function == 'contrastive':
            TP = jnp.sum(y_binary * total_resp)
            FP = jnp.sum((1.0 - y_binary) * total_resp)
            FN = jnp.sum(y_binary) - TP
            TN = jnp.sum(1.0 - y_binary) - FP
            C = 1.0 - (TP * TN - FP * FN) / (
                (TP + FP + 1e-10) * (TN + FN + 1e-10)
            )
        elif self.cost_function == 'brier':
            C = jnp.mean((y_binary - total_resp) ** 2)
        else:
            C = -jnp.mean(
                y_binary * jnp.log(total_resp) +
                (1.0 - y_binary) * jnp.log(1.0 - total_resp)
            )

        return self.alpha * R + (1.0 - self.alpha) * C

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omega_ = params['omega']

    def decision_function(self, X):
        """Compute scores using Omega-projected distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('nkd,dl->nkl', diff, self.omega_)
        sq_distances = jnp.sum(projected ** 2, axis=2)
        n_protos = self.prototypes_.shape[0]

        if self.response_type == 'gaussian':
            p_k = jax.nn.softmax(-self.gamma_resp * sq_distances, axis=1)
        elif self.response_type == 'student_t':
            p_unnorm = (
                (1.0 + sq_distances / self.nu) ** (-(self.nu + 1) / 2)
            )
            p_k = p_unnorm / (
                jnp.sum(p_unnorm, axis=1, keepdims=True) + 1e-10
            )
        else:
            p_k = jnp.ones_like(sq_distances) / n_protos

        heaviside = jax.nn.sigmoid(
            (self.thetas_[None, :] - sq_distances) / (self.sigma + 1e-10)
        )
        return jnp.clip(jnp.sum(p_k * heaviside, axis=1), 0.0, 1.0)

    @property
    def omega_matrix(self):
        """Return the learned projection matrix Ω."""
        if self.omega_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.omega_

    @property
    def lambda_matrix(self):
        """Return the implicit metric Λ = Ω^T Ω."""
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
        hp['latent_dim'] = self.latent_dim
        return hp
