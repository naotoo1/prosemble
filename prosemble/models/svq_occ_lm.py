"""
Local Matrix SVQ-OCC (SVQ-OCC-LM).

Extends SVQ-OCC with per-prototype Omega matrices (LGMLVQ-style).
Each prototype w_k learns its own projection Ω_k:

    d_{Ω_k}(x, w_k) = ||Ω_k(x - w_k)||²

This allows each prototype to focus on different feature subspaces,
enabling more flexible decision boundaries for one-class classification.

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


class SVQOCC_LM(SVQOCC):
    """Local Matrix SVQ-OCC with per-prototype Omega projections.

    Each prototype learns its own local metric via Ω_k, allowing
    different prototypes to attend to different feature subspaces.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of each projected space. Default: n_features.
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
    omegas_ : array of shape (n_prototypes, n_features, latent_dim)
        Learned per-prototype projection matrices.
    """

    def __init__(self, latent_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.omegas_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['omegas'] = self.omegas_
        return params

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)
        n_features = X.shape[1]
        n_protos = self.n_prototypes
        latent_dim = self.latent_dim if self.latent_dim is not None else n_features
        omega_single = identity_omega_init(n_features, latent_dim)
        params['omegas'] = jnp.tile(omega_single[None, :, :], (n_protos, 1, 1))
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
        omegas = params['omegas']

        n_protos = prototypes.shape[0]

        # Per-prototype Omega-projected distances
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, K, d)
        projected = jnp.einsum('nkd,kdl->nkl', diff, omegas)  # (n, K, l)
        sq_distances = jnp.sum(projected ** 2, axis=2)  # (n, K)

        target_mask = (y == self._target_label)

        # Representation cost R
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
        self.omegas_ = params['omegas']

    def decision_function(self, X):
        """Compute scores using per-prototype Omega-projected distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('nkd,kdl->nkl', diff, self.omegas_)
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
        hp['latent_dim'] = self.latent_dim
        return hp
