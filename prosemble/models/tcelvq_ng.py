"""
Tangent Cross-Entropy LVQ with Neural Gas cooperation (TCELVQ-NG).

Combines CELVQ-NG's cross-entropy loss over NG-weighted softmax logits
with per-prototype tangent subspaces. Each prototype has an orthonormal
basis Omega_k defining invariance directions; the tangent distance
measures distance in the orthogonal complement:
    d(x, w_k) = ||(I - Omega_k Omega_k^T)(x - w_k)||^2

Neural Gas rank-based cooperation ensures robust prototype placement
while cross-entropy over all classes provides gradient flow to all
tangent subspaces simultaneously.

When gamma -> 0, only the nearest prototype per class dominates and
TCELVQ-NG recovers a tangent variant of standard CELVQ.

References
----------
.. [1] Villmann, T., et al. (2019). Analysis of variants of
       classification learning vector quantization by a stochastic
       setting.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
.. [3] Saralajew, S., & Villmann, T. (2016). Adaptive tangent
       distances in generalized learning vector quantization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.initializers import random_omega_init
from prosemble.core.utils import orthogonalize


@jit
def _predict_tcelvq_ng_jit(X, prototypes, omegas, proto_labels):
    """JIT-compiled TCELVQ-NG prediction with tangent distance."""
    diff = X[:, None, :] - prototypes[None, :, :]
    proj = jnp.einsum('npd,pds->nps', diff, omegas)
    recon = jnp.einsum('nps,pds->npd', proj, omegas)
    tang_diff = diff - recon
    distances = jnp.sum(tang_diff ** 2, axis=2)
    return wtac(distances, proto_labels)


class TCELVQ_NG(SupervisedPrototypeModel):
    """Tangent Cross-Entropy LVQ with Neural Gas neighborhood cooperation.

    Combines three key ideas:
    - Cross-entropy loss: softmax over all-class NG-weighted distances
    - Neural Gas cooperation: all same-class prototypes participate,
      weighted by rank via exp(-rank / gamma)
    - Tangent subspaces: d(x, w_k) = ||(I - Omega_k Omega_k^T)(x - w_k)||^2
      measures distance in the orthogonal complement of each prototype's
      learned invariance subspace

    The neighborhood range gamma decays during training from gamma_init
    to gamma_final. When gamma -> 0, TCELVQ-NG recovers a tangent CELVQ.

    Parameters
    ----------
    subspace_dim : int
        Dimension of each prototype's tangent subspace.
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
    omegas_ : array of shape (n_prototypes, n_features, subspace_dim)
        Learned tangent subspace bases after training.
    gamma_ : float
        Final gamma value after training.
    """

    def __init__(self, subspace_dim=2, gamma_init=None, gamma_final=0.01,
                 gamma_decay=None, **kwargs):
        super().__init__(**kwargs)
        self.subspace_dim = subspace_dim
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
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        n_protos = prototypes.shape[0]

        # Initialize each Omega as random orthogonal
        keys = jax.random.split(key2, n_protos)
        omegas = jnp.stack([
            random_omega_init(n_features, self.subspace_dim, k) for k in keys
        ])  # (p, d, subspace_dim)

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
        omegas = params['omegas']  # (p, d, s)
        gamma = params['gamma']
        n_classes = self.n_classes_

        # 1. Tangent distance: d(x, w_k) = ||(I - Omega_k Omega_k^T)(x - w_k)||^2
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        proj_onto_subspace = jnp.einsum('npd,pds->nps', diff, omegas)  # (n, p, s)
        reconstruction = jnp.einsum('nps,pds->npd', proj_onto_subspace, omegas)  # (n, p, d)
        tangent_diff = diff - reconstruction  # (n, p, d)
        distances = jnp.sum(tangent_diff ** 2, axis=2)  # (n, p)

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
        # Decay gamma AND re-orthogonalize tangent bases
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        omegas = jax.vmap(orthogonalize)(params['omegas'])
        return {**params, 'gamma': new_gamma, 'omegas': omegas}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omegas_ = params['omegas']
        self.gamma_ = float(params['gamma'])

    def predict(self, X):
        """Predict using tangent distance."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_tcelvq_ng_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_
        )

    def predict_proba(self, X):
        """Predict calibrated probabilities using tangent distances.

        Uses per-class min pooling with tangent distance, consistent
        with the training objective.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        # Compute tangent distances
        diff = X[:, None, :] - self.prototypes_[None, :, :]
        proj = jnp.einsum('npd,pds->nps', diff, self.omegas_)
        recon = jnp.einsum('nps,pds->npd', proj, self.omegas_)
        tang_diff = diff - recon
        distances = jnp.sum(tang_diff ** 2, axis=2)

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
        hp['subspace_dim'] = self.subspace_dim
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        return hp
