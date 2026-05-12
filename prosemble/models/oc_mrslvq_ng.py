"""
One-Class Matrix RSLVQ with Neural Gas cooperation (OC-MRSLVQ-NG)
and One-Class Local Matrix RSLVQ with NG (OC-LMRSLVQ-NG).

Combines OC-MRSLVQ/OC-LMRSLVQ's Omega metric adaptation and Gaussian
soft-weighting with Neural Gas rank-based neighborhood cooperation:

    p(k|x) = exp(-d_Ω_k / 2σ²) / Σ_j exp(-d_Ω_j / 2σ²)
    h_k = exp(-rank_k / γ)
    w_k = p(k|x) · h_k / Σ_j p(j|x) · h_j
    loss = mean(Σ_k w_k · sigmoid(μ_k + margin, β))

OC-MRSLVQ-NG: global Omega projection matrix (same for all prototypes).
OC-LMRSLVQ-NG: per-prototype Omega_k matrices (local metric adaptation).

References
----------
.. [1] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization. Neural
       Computation.
.. [2] Seo, S., & Obermayer, K. (2007). Soft Learning Vector
       Quantization. Neural Computation, 19(6):1589-1604.
.. [3] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
"""

import jax.numpy as jnp

from prosemble.models.oc_rslvq_ng_mixin import OCRSLVQNGMixin
from prosemble.models.oc_mrslvq import OCMRSLVQ, OCLMRSLVQ


class OCMRSLVQ_NG(OCRSLVQNGMixin, OCMRSLVQ):
    """One-Class Matrix RSLVQ with Neural Gas neighborhood cooperation.

    Learns a global Omega projection with combined Gaussian + NG
    rank-weighted loss for one-class classification.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture for prototype weighting.
    latent_dim : int, optional
        Dimensionality of the projected space. Default: n_features.
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.
    gamma_init : float, optional
        Initial neighborhood range. Default: n_prototypes / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.

    Attributes
    ----------
    omega_ : array of shape (n_features, latent_dim)
        Learned projection matrix.
    thetas_ : array of shape (n_prototypes,)
        Learned per-prototype acceptance thresholds.
    gamma_ : float
        Final gamma value after training.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def _compute_distances(self, params, X):
        diff = X[:, None, :] - params['prototypes'][None, :, :]
        projected = jnp.einsum('nkd,dl->nkl', diff, params['omega'])
        return jnp.sum(projected ** 2, axis=2)

    def _inference_distances(self, X):
        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('nkd,dl->nkl', diff, self.omega_)
        return jnp.sum(projected ** 2, axis=2)


class OCLMRSLVQ_NG(OCRSLVQNGMixin, OCLMRSLVQ):
    """One-Class Local Matrix RSLVQ with Neural Gas neighborhood cooperation.

    Learns per-prototype Omega_k projection matrices with combined
    Gaussian + NG rank-weighted loss for one-class classification.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture for prototype weighting.
    latent_dim : int, optional
        Latent space dimensionality per prototype. Default: n_features.
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.
    gamma_init : float, optional
        Initial neighborhood range. Default: n_prototypes / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.

    Attributes
    ----------
    omegas_ : array of shape (n_prototypes, n_features, latent_dim)
        Learned per-prototype projection matrices.
    thetas_ : array of shape (n_prototypes,)
        Learned per-prototype acceptance thresholds.
    gamma_ : float
        Final gamma value after training.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def _compute_distances(self, params, X):
        diff = X[:, None, :] - params['prototypes'][None, :, :]
        projected = jnp.einsum('nkd,kdl->nkl', diff, params['omegas'])
        return jnp.sum(projected ** 2, axis=2)

    def _inference_distances(self, X):
        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('nkd,kdl->nkl', diff, self.omegas_)
        return jnp.sum(projected ** 2, axis=2)
