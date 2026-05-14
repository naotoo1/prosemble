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

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters (optimizer,
        distance_fn, lr_scheduler, callbacks, patience, etc.).
    """

    def __init__(self, beta=10.0, n_prototypes_per_class=1, max_iter=100,
                 lr=0.01, epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None, mixed_precision=None):
        super().__init__(
            n_prototypes_per_class=n_prototypes_per_class,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, distance_fn=distance_fn,
            optimizer=optimizer, transfer_fn=transfer_fn, margin=margin,
            callbacks=callbacks, use_scan=use_scan, batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            prototypes_initializer=prototypes_initializer,
            patience=patience, restore_best=restore_best,
            class_weight=class_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ema_decay=ema_decay, freeze_params=freeze_params,
            lookahead=lookahead, mixed_precision=mixed_precision,
        )
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
        return lvq1_loss(distances, y, proto_labels)


class GLVQ21(SupervisedPrototypeModel):
    """GLVQ with LVQ2.1-style loss (gradient-based, unnormalized).

    Loss: d+ - d- (no normalization by d+ + d-).

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
        return lvq21_loss(distances, y, proto_labels)
