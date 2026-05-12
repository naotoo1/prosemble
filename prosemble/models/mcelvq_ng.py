"""
Matrix Cross-Entropy LVQ with Neural Gas cooperation (MCELVQ-NG).

Combines CELVQ-NG's cross-entropy loss over NG-weighted softmax logits
with a global linear transformation Omega that learns a discriminative
subspace: d(x, w) = ||Omega(x - w)||^2.

The Omega matrix captures feature correlations and projects data into
a space where cross-entropy classification is more effective. Neural Gas
rank-based cooperation ensures robust prototype placement.

When gamma -> 0, only the nearest prototype per class dominates and
MCELVQ-NG recovers a matrix variant of standard CELVQ.

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
def _predict_mcelvq_ng_jit(X, prototypes, omega, proto_labels):
    """JIT-compiled MCELVQ-NG prediction with learned Omega metric."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,dl->npl', diff, omega)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


class MCELVQ_NG(CELVQNGMixin, CELVQ):
    """Matrix Cross-Entropy LVQ with Neural Gas neighborhood cooperation.

    Combines three key ideas:

    - Cross-entropy loss: softmax over all-class NG-weighted distances
    - Neural Gas cooperation: all same-class prototypes participate,
      weighted by rank via ``exp(-rank / gamma)``
    - Global Omega projection: ``d(x, w) = ||Omega(x - w)||^2`` learns
      feature correlations and a discriminative subspace

    The neighborhood range gamma decays during training from gamma_init
    to gamma_final. When gamma -> 0, MCELVQ-NG recovers a matrix CELVQ.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of the latent space. If None, uses input dim.
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
    omega_ : array
        Learned Omega projection matrix after training.
    gamma_ : float
        Final gamma value after training.
    """

    def __init__(self, latent_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.omega_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['omega'] = self.omega_
        return params

    def _init_metric_params(self, params, X, prototypes, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        omega = identity_omega_init(n_features, latent_dim)
        params['omega'] = omega
        return params

    def _compute_distances(self, params, X):
        diff = X[:, None, :] - params['prototypes'][None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,dl->npl', diff, params['omega'])  # (n, p, l)
        return jnp.sum(projected ** 2, axis=2)  # (n, p)

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omega_ = params['omega']

    @property
    def omega_matrix(self):
        """Return the learned Omega matrix."""
        if self.omega_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_

    @property
    def lambda_matrix(self):
        """Return Lambda = Omega^T Omega (relevance matrix)."""
        if self.omega_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_.T @ self.omega_

    def predict(self, X):
        """Predict using learned Omega distance."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_mcelvq_ng_jit(
            X, self.prototypes_, self.omega_, self.prototype_labels_
        )

    def predict_proba(self, X):
        """Predict calibrated probabilities using Omega-transformed distances.

        Uses NG-weighted pooling with the learned Omega metric, matching
        the training objective exactly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        # Compute Omega-transformed distances
        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('npd,dl->npl', diff, self.omega_)
        distances = jnp.sum(projected ** 2, axis=2)

        # Softmax over per-class min distances (consistent with CELVQ)
        from prosemble.core.pooling import stratified_min_pooling
        class_dists = stratified_min_pooling(
            distances, self.prototype_labels_, self.n_classes_
        )
        return jax.nn.softmax(-class_dists, axis=1)

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
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
