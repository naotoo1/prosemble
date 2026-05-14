"""
Localized Matrix Cross-Entropy LVQ with Neural Gas cooperation (LCELVQ-NG).

Combines CELVQ-NG's cross-entropy loss over NG-weighted softmax logits
with per-prototype Omega matrices that learn local metrics adapted to
each prototype's region: d(x, w_k) = ||Omega_k(x - w_k)||^2.

Each prototype learns its own discriminative subspace while Neural Gas
rank-based cooperation ensures robust prototype placement. The cross-
entropy loss over all classes simultaneously provides gradient flow to
all local matrices.

When gamma -> 0, only the nearest prototype per class dominates and
LCELVQ-NG recovers a localized matrix variant of standard CELVQ.

References
----------
.. [1] Villmann, T., et al. (2019). Analysis of variants of
       classification learning vector quantization by a stochastic
       setting.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
.. [3] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from prosemble.models.celvq_ng_mixin import CELVQNGMixin
from prosemble.models.crossentropy_lvq import CELVQ
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init


@jit
def _predict_lcelvq_ng_jit(X, prototypes, omegas, proto_labels):
    """JIT-compiled LCELVQ-NG prediction with per-prototype Omega metrics."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,pdl->npl', diff, omegas)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


class LCELVQ_NG(CELVQNGMixin, CELVQ):
    """Localized Matrix Cross-Entropy LVQ with Neural Gas cooperation.

    Combines three key ideas:

    - Cross-entropy loss: softmax over all-class NG-weighted distances
    - Neural Gas cooperation: all same-class prototypes participate,
      weighted by rank via ``exp(-rank / gamma)``
    - Per-prototype Omega_k: ``d(x, w_k) = ||Omega_k(x - w_k)||^2`` learns
      local metrics adapted to each prototype's region

    The neighborhood range gamma decays during training from gamma_init
    to gamma_final. When gamma -> 0, LCELVQ-NG recovers a localized
    matrix CELVQ.

    Parameters
    ----------
    latent_dim : int, optional
        Latent space dimensionality per prototype. If None, uses input dim.
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    gamma_init : float, optional
        Initial neighborhood range. Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay factor for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, latent_dim=None, gamma_init=None,
                 gamma_final=0.01, gamma_decay=None,
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
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
            gamma_init=gamma_init, gamma_final=gamma_final,
            gamma_decay=gamma_decay,
            n_prototypes_per_class=n_prototypes_per_class,
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
        self.omegas_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['omegas'] = self.omegas_
        return params

    def _init_metric_params(self, params, X, prototypes, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        n_protos = prototypes.shape[0]
        omega_single = identity_omega_init(n_features, latent_dim)
        omegas = jnp.tile(omega_single[None, :, :], (n_protos, 1, 1))
        params['omegas'] = omegas
        return params

    def _compute_distances(self, params, X):
        diff = X[:, None, :] - params['prototypes'][None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,pdl->npl', diff, params['omegas'])  # (n, p, l)
        return jnp.sum(projected ** 2, axis=2)  # (n, p)

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omegas_ = params['omegas']

    def predict(self, X):
        """Predict using per-prototype Omega distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_lcelvq_ng_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_
        )

    def predict_proba(self, X):
        """Predict calibrated probabilities using per-prototype Omega distances.

        Uses per-class min pooling with the learned local Omega metrics,
        consistent with the training objective.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        # Compute per-prototype Omega-transformed distances
        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('npd,pdl->npl', diff, self.omegas_)
        distances = jnp.sum(projected ** 2, axis=2)

        # Softmax over per-class min distances
        from prosemble.core.pooling import stratified_min_pooling
        class_dists = stratified_min_pooling(
            distances, self.prototype_labels_, self.n_classes_
        )
        return jax.nn.softmax(-class_dists, axis=1)

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
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
