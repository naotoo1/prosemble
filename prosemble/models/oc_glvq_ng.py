"""
One-Class GLVQ with Neural Gas cooperation (OC-GLVQ-NG).

Extends OC-GLVQ with Neural Gas neighborhood cooperation. Instead of
only the nearest prototype contributing to the loss, ALL prototypes
participate weighted by their distance rank.

References
----------
.. [1] Martinetz, T., Berkovich, S., & Schulten, K. (1993).
       Neural-gas network for the quantization of continuous input
       spaces. IEEE Transactions on Neural Networks.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
"""

from prosemble.models.oc_glvq_ng_mixin import NGCooperationMixin
from prosemble.models.oc_glvq import OCGLVQ


class OCGLVQ_NG(NGCooperationMixin, OCGLVQ):
    """One-Class GLVQ with Neural Gas neighborhood cooperation.

    All prototypes participate in the loss, weighted by their distance
    rank via exp(-rank / gamma). Uses squared Euclidean distance.

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
    gamma_ : float
        Final gamma value after training.
    """

    def _compute_distances(self, params, X):
        return self.distance_fn(X, params['prototypes'])
