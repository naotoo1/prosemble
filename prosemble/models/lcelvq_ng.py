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

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init


@jit
def _predict_lcelvq_ng_jit(X, prototypes, omegas, proto_labels):
    """JIT-compiled LCELVQ-NG prediction with per-prototype Omega metrics."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,pdl->npl', diff, omegas)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


class LCELVQ_NG(SupervisedPrototypeModel):
    """Localized Matrix Cross-Entropy LVQ with Neural Gas cooperation.

    Combines three key ideas:
    - Cross-entropy loss: softmax over all-class NG-weighted distances
    - Neural Gas cooperation: all same-class prototypes participate,
      weighted by rank via exp(-rank / gamma)
    - Per-prototype Omega_k: d(x, w_k) = ||Omega_k(x - w_k)||^2 learns
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

    Attributes
    ----------
    omegas_ : array of shape (n_prototypes, n_features, latent_dim)
        Learned per-prototype Omega matrices after training.
    gamma_ : float
        Final gamma value after training.
    """

    def __init__(self, latent_dim=None, gamma_init=None, gamma_final=0.01,
                 gamma_decay=None, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.omegas_ = None
        self.gamma_ = None

        # Ensure gamma is frozen from optimizer (not trainable)
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        params['omegas'] = self.omegas_
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        n_protos = prototypes.shape[0]
        # Each prototype gets its own Omega
        omega_single = identity_omega_init(n_features, latent_dim)
        omegas = jnp.tile(omega_single[None, :, :], (n_protos, 1, 1))

        # Compute gamma_init from prototype count if not set
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)
        gamma_init = self.gamma_init if self.gamma_init is not None else max_per_class / 2.0
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        # Compute decay factor
        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / self.max_iter)

        params = {
            'prototypes': prototypes,
            'omegas': omegas,
            'gamma': jnp.array(gamma_init, dtype=jnp.float32),
        }
        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        omegas = params['omegas']  # (p, d, l)
        gamma = params['gamma']
        n_classes = self.n_classes_

        # 1. Per-prototype Omega distance: d(x, w_k) = ||Omega_k(x - w_k)||^2
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,pdl->npl', diff, omegas)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)

        # 2. NG-weighted per-class distance pooling
        INF = jnp.finfo(distances.dtype).max
        class_dists_list = []

        for c in range(n_classes):
            # Mask: which prototypes belong to class c
            class_mask = (proto_labels == c)  # (p,)

            # Distances to class c prototypes (INF for non-class)
            d_class = jnp.where(class_mask[None, :], distances, INF)  # (n, p)

            # Rank within class c (double argsort)
            order = jnp.argsort(d_class, axis=1)
            ranks = jnp.argsort(order, axis=1).astype(jnp.float32)  # (n, p)

            # NG neighborhood function
            h = jnp.exp(-ranks / (gamma + 1e-10))  # (n, p)
            h = jnp.where(class_mask[None, :], h, 0.0)  # zero non-class

            # Normalize within class
            C = jnp.sum(h, axis=1, keepdims=True)  # (n, 1)
            h_normalized = h / (C + 1e-10)  # (n, p)

            # NG-weighted class distance
            weighted_dist = jnp.sum(h_normalized * distances, axis=1)  # (n,)
            class_dists_list.append(weighted_dist)

        # 3. Stack into (n, n_classes)
        class_dists = jnp.stack(class_dists_list, axis=1)  # (n, n_classes)

        # 4. Cross-entropy loss
        logits = -class_dists  # negate: smaller distance = larger logit
        log_probs = jax.nn.log_softmax(logits, axis=1)
        target_one_hot = jax.nn.one_hot(y, n_classes)
        return -jnp.mean(jnp.sum(target_one_hot * log_probs, axis=1))

    def _post_update(self, params):
        # Decay gamma (neighborhood range) each step
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omegas_ = params['omegas']
        self.gamma_ = float(params['gamma'])

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
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omegas_' in arrays:
            self.omegas_ = jnp.asarray(arrays['omegas_'])
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
