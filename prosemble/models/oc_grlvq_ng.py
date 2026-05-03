"""
One-Class GRLVQ with Neural Gas cooperation (OC-GRLVQ-NG).

Combines OC-GRLVQ's per-feature relevance weighting with Neural Gas
neighborhood cooperation.

References
----------
.. [1] Hammer, Villmann (2002). Generalized Relevance Learning Vector
       Quantization. Neural Networks, 15(8-9), 1059-1068.
.. [2] Hammer, Strickert, Villmann (2003). Supervised Neural Gas with
       General Similarity Measure. Neural Processing Letters.
"""

import jax
import jax.numpy as jnp

from prosemble.models.oc_glvq_ng_mixin import NGCooperationMixin
from prosemble.models.oc_grlvq import OCGRLVQ


class OCGRLVQ_NG(NGCooperationMixin, OCGRLVQ):
    """One-Class GRLVQ with Neural Gas neighborhood cooperation.

    Learns per-feature relevance weights with NG rank-weighted loss.

    Parameters
    ----------
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
    relevances_ : array of shape (n_features,)
        Learned per-feature relevance weights.
    gamma_ : float
        Final gamma value after training.
    """

    def _compute_distances(self, params, X):
        lam = jax.nn.softmax(params['relevances'])
        diff = X[:, None, :] - params['prototypes'][None, :, :]
        return jnp.sum(lam[None, None, :] * diff ** 2, axis=2)
