"""
One-Class LGMLVQ with Neural Gas cooperation (OC-LGMLVQ-NG).

Combines OC-LGMLVQ's per-prototype Omega matrices with Neural Gas
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
from prosemble.models.oc_lgmlvq import OCLGMLVQ


class OCLGMLVQ_NG(NGCooperationMixin, OCLGMLVQ):
    """One-Class LGMLVQ with Neural Gas neighborhood cooperation.

    Learns per-prototype local Omega projections with NG rank-weighted loss.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of each projected space. Default: n_features.
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
