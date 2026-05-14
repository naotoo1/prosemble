"""
One-Class GTLVQ (OC-GTLVQ).

Extends OC-GLVQ with per-prototype tangent subspaces (GTLVQ-style).
Each prototype w_k learns an orthonormal basis Ω_k defining a local
invariance subspace. The tangent distance measures the orthogonal
complement:

    d_T(x, w_k) = ||(I - Ω_k Ω_k^T)(x - w_k)||²

References
----------
.. [1] Sato, A., & Yamada, K. (1995). Generalized Learning Vector
       Quantization. NIPS.
.. [2] Saralajew, Villmann (2016). Adaptive tangent distances in
       generalized learning vector quantization. WSOM 2016.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.oc_glvq import OCGLVQ
from prosemble.core.initializers import random_omega_init
from prosemble.core.utils import orthogonalize
from prosemble.core.activations import sigmoid_beta


class OCGTLVQ(OCGLVQ):
    """One-Class GTLVQ with per-prototype tangent subspaces.

    Each prototype learns an orthonormal basis Ω_k that defines
    directions of local invariance. Only the distance orthogonal
    to this subspace is used for classification.

    Parameters
    ----------
    subspace_dim : int
        Dimensionality of each tangent subspace. Default: 2.
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.

    Attributes
    ----------
    omegas_ : array of shape (n_prototypes, n_features, subspace_dim)
        Learned per-prototype orthonormal tangent bases.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, subspace_dim=2, n_prototypes=3, target_label=None,
                 beta=10.0, max_iter=100, lr=0.01, epsilon=1e-6,
                 random_seed=42, distance_fn=None, optimizer='adam',
                 transfer_fn=None, margin=0.0, callbacks=None,
                 use_scan=True, batch_size=None, lr_scheduler=None,
                 lr_scheduler_kwargs=None, prototypes_initializer=None,
                 patience=None, restore_best=False, class_weight=None,
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
        self.subspace_dim = subspace_dim
        self.omegas_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['omegas'] = self.omegas_
        return params

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)
        n_features = X.shape[1]
        n_protos = self.n_prototypes
        key1, key2 = jax.random.split(self.key, 2)
        keys = jax.random.split(key2, n_protos)
        params['omegas'] = jnp.stack([
            random_omega_init(n_features, self.subspace_dim, k) for k in keys
        ])
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
        omegas = params['omegas']

        # Tangent distance: ||(I - Ω_k Ω_k^T)(x - w_k)||²
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, K, d)
        proj = jnp.einsum('nkd,kds->nks', diff, omegas)  # (n, K, s)
        recon = jnp.einsum('nks,kds->nkd', proj, omegas)  # (n, K, d)
        tang_diff = diff - recon  # orthogonal complement
        distances = jnp.sum(tang_diff ** 2, axis=2)  # (n, K)

        # OC-GLVQ mu
        n = X.shape[0]
        nearest_idx = jnp.argmin(distances, axis=1)
        d_nearest = distances[jnp.arange(n), nearest_idx]
        theta_nearest = thetas[nearest_idx]
        s = jnp.where(y == self._target_label, 1.0, -1.0)
        mu = s * (d_nearest - theta_nearest) / (d_nearest + theta_nearest + 1e-10)
        transfer = self.transfer_fn or sigmoid_beta
        return jnp.mean(transfer(mu + self.margin, self.beta))

    def _post_update(self, params):
        params = super()._post_update(params)
        omegas = jax.vmap(orthogonalize)(params['omegas'])
        return {**params, 'omegas': omegas}

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omegas_ = params['omegas']

    def decision_function(self, X):
        """Compute scores using tangent distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        diff = X[:, None, :] - self.prototypes_[None, :, :]
        proj = jnp.einsum('nkd,kds->nks', diff, self.omegas_)
        recon = jnp.einsum('nks,kds->nkd', proj, self.omegas_)
        tang_diff = diff - recon
        distances = jnp.sum(tang_diff ** 2, axis=2)
        n = X.shape[0]
        nearest_idx = jnp.argmin(distances, axis=1)
        d_nearest = distances[jnp.arange(n), nearest_idx]
        theta_nearest = self.thetas_[nearest_idx]
        mu = (d_nearest - theta_nearest) / (d_nearest + theta_nearest + 1e-10)
        return 1.0 - jax.nn.sigmoid(self.beta * mu)

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
        hp['subspace_dim'] = self.subspace_dim
        return hp
