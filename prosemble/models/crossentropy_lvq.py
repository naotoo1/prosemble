"""
Cross-Entropy LVQ (CELVQ).

Uses cross-entropy loss on softmin of per-class minimum distances
instead of the GLVQ relative distance difference.

References
----------
.. [1] Villmann, T., et al. (2019). Analysis of variants of
       classification learning vector quantization by a stochastic
       setting.
"""

import jax.numpy as jnp

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.losses import cross_entropy_lvq_loss


class CELVQ(SupervisedPrototypeModel):
    """Cross-Entropy Learning Vector Quantization.

    Computes per-class minimum distances, negates them to get logits,
    then applies cross-entropy loss against true labels.

    Parameters
    ----------
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def _compute_loss(self, params, X, y, proto_labels):
        distances = self.distance_fn(X, params['prototypes'])
        return cross_entropy_lvq_loss(distances, y, proto_labels, self.n_classes_)
