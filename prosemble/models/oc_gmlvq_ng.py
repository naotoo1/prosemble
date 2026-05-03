"""
One-Class GMLVQ with Neural Gas cooperation (OC-GMLVQ-NG).

Combines OC-GMLVQ's global Omega projection with Neural Gas
neighborhood cooperation.

References
----------
.. [1] Schneider, Biehl, Hammer (2009). Adaptive Relevance Matrices
       in Learning Vector Quantization. Neural Computation, 21(12).
.. [2] Hammer, Strickert, Villmann (2003). Supervised Neural Gas with
       General Similarity Measure. Neural Processing Letters.
"""

import jax.numpy as jnp

from prosemble.models.oc_glvq_ng_mixin import NGCooperationMixin
from prosemble.models.oc_gmlvq import OCGMLVQ


class OCGMLVQ_NG(NGCooperationMixin, OCGMLVQ):
    """One-Class GMLVQ with Neural Gas neighborhood cooperation.

    Learns a global Omega projection with NG rank-weighted loss.

    Parameters
    ----------
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
    gamma_ : float
        Final gamma value after training.
    """

    def _compute_distances(self, params, X):
        diff = X[:, None, :] - params['prototypes'][None, :, :]
        projected = jnp.einsum('nkd,dl->nkl', diff, params['omega'])
        return jnp.sum(projected ** 2, axis=2)
