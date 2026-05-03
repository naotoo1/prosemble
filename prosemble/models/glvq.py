"""
Generalized Learning Vector Quantization (GLVQ) and variants.

Implements GLVQ, GLVQ1, and GLVQ21 — all gradient-based supervised
prototype models using the relative distance difference loss.

References
----------
.. [1] Sato, A., & Yamada, K. (1995). Generalized Learning Vector
       Quantization. NIPS.
"""

import jax
import jax.numpy as jnp

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.losses import glvq_loss, glvq_loss_with_transfer, lvq1_loss, lvq21_loss


class GLVQ(SupervisedPrototypeModel):
    """Generalized Learning Vector Quantization.

    Loss: mu = (d+ - d-) / (d+ + d-), with optional transfer function.

    Parameters
    ----------
    n_prototypes_per_class : int
        Prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    transfer_fn : callable, optional
        Transfer/activation function for loss shaping.
    margin : float
        Margin added to mu before transfer.
    beta : float
        Parameter for transfer function (e.g., sigmoid steepness).
    optimizer : str or optax optimizer
        Optimizer ('adam', 'sgd', or optax object).
    """

    def __init__(self, beta=10.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def _compute_loss(self, params, X, y, proto_labels):
        distances = self.distance_fn(X, params['prototypes'])
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        return hp


class GLVQ1(SupervisedPrototypeModel):
    """GLVQ with LVQ1-style loss (gradient-based).

    Loss: d+ when correct, -d- when wrong.
    """

    def _compute_loss(self, params, X, y, proto_labels):
        distances = self.distance_fn(X, params['prototypes'])
        return lvq1_loss(distances, y, proto_labels)


class GLVQ21(SupervisedPrototypeModel):
    """GLVQ with LVQ2.1-style loss (gradient-based, unnormalized).

    Loss: d+ - d- (no normalization by d+ + d-).
    """

    def _compute_loss(self, params, X, y, proto_labels):
        distances = self.distance_fn(X, params['prototypes'])
        return lvq21_loss(distances, y, proto_labels)
