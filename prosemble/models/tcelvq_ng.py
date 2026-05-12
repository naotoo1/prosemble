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

from prosemble.models.celvq_ng_mixin import CELVQNGMixin
from prosemble.models.crossentropy_lvq import CELVQ
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


class TCELVQ_NG(CELVQNGMixin, CELVQ):
    """Tangent Cross-Entropy LVQ with Neural Gas neighborhood cooperation.

    Combines three key ideas:

    - Cross-entropy loss: softmax over all-class NG-weighted distances
    - Neural Gas cooperation: all same-class prototypes participate,
      weighted by rank via ``exp(-rank / gamma)``
    - Tangent subspaces: ``d(x, w_k) = ||(I - Omega_k Omega_k^T)(x - w_k)||^2``
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

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, subspace_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.subspace_dim = subspace_dim
        self.omegas_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['omegas'] = self.omegas_
        return params

    def _init_metric_params(self, params, X, prototypes, key):
        n_features = X.shape[1]
        n_protos = prototypes.shape[0]
        keys = jax.random.split(key, n_protos)
        omegas = jnp.stack([
            random_omega_init(n_features, self.subspace_dim, k) for k in keys
        ])  # (p, d, subspace_dim)
        params['omegas'] = omegas
        return params

    def _compute_distances(self, params, X):
        diff = X[:, None, :] - params['prototypes'][None, :, :]  # (n, p, d)
        proj_onto_subspace = jnp.einsum('npd,pds->nps', diff, params['omegas'])  # (n, p, s)
        reconstruction = jnp.einsum('nps,pds->npd', proj_onto_subspace, params['omegas'])  # (n, p, d)
        tangent_diff = diff - reconstruction  # (n, p, d)
        return jnp.sum(tangent_diff ** 2, axis=2)  # (n, p)

    def _post_update(self, params):
        # Decay gamma AND re-orthogonalize tangent bases
        params = super()._post_update(params)
        omegas = jax.vmap(orthogonalize)(params['omegas'])
        return {**params, 'omegas': omegas}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omegas_ = params['omegas']

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
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omegas_' in arrays:
            self.omegas_ = jnp.asarray(arrays['omegas_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['subspace_dim'] = self.subspace_dim
        return hp
