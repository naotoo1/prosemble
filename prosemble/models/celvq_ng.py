"""
Cross-Entropy LVQ with Neural Gas cooperation (CELVQ-NG).

Combines CELVQ's cross-entropy loss over all-class softmax logits with
Neural Gas rank-based neighborhood cooperation. Instead of using the
hard per-class minimum distance (as in CELVQ), prototypes within each
class are weighted by their NG rank: h_k = exp(-rank / gamma). This
replaces the hard min pooling with a soft NG-weighted pooling.

When gamma -> 0, only the nearest prototype per class dominates and
CELVQ-NG recovers standard CELVQ.

References
----------
.. [1] Villmann, T., et al. (2019). Analysis of variants of
       classification learning vector quantization by a stochastic
       setting.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
"""

from prosemble.models.celvq_ng_mixin import CELVQNGMixin
from prosemble.models.crossentropy_lvq import CELVQ


class CELVQ_NG(CELVQNGMixin, CELVQ):
    """Cross-Entropy LVQ with Neural Gas neighborhood cooperation.

    For each class, prototypes are ranked by distance and weighted
    by exp(-rank / gamma). The NG-weighted class distances become
    logits for cross-entropy loss over all classes simultaneously.

    Parameters
    ----------
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
    gamma_ : float
        Final gamma value after training.
    """

    def _compute_distances(self, params, X):
        return self.distance_fn(X, params['prototypes'])
