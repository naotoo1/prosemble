"""
One-Class GTLVQ with Neural Gas cooperation (OC-GTLVQ-NG).

Combines OC-GTLVQ's per-prototype tangent subspaces with Neural Gas
neighborhood cooperation.

References
----------
.. [1] Saralajew, Villmann (2016). Adaptive tangent distances in
       generalized learning vector quantization. WSOM 2016.
.. [2] Hammer, Strickert, Villmann (2003). Supervised Neural Gas with
       General Similarity Measure. Neural Processing Letters.
"""

import jax.numpy as jnp

from prosemble.models.oc_glvq_ng_mixin import NGCooperationMixin
from prosemble.models.oc_gtlvq import OCGTLVQ


class OCGTLVQ_NG(NGCooperationMixin, OCGTLVQ):
    """One-Class GTLVQ with Neural Gas neighborhood cooperation.

    Learns per-prototype tangent subspaces with NG rank-weighted loss.

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
    gamma_init : float, optional
        Initial neighborhood range. Default: n_prototypes / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.

    Attributes
    ----------
    omegas_ : array of shape (n_prototypes, n_features, subspace_dim)
        Learned per-prototype orthonormal tangent bases.
    gamma_ : float
        Final gamma value after training.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def _compute_distances(self, params, X):
        diff = X[:, None, :] - params['prototypes'][None, :, :]
        proj = jnp.einsum('nkd,kds->nks', diff, params['omegas'])
        recon = jnp.einsum('nks,kds->nkd', proj, params['omegas'])
        return jnp.sum((diff - recon) ** 2, axis=2)
