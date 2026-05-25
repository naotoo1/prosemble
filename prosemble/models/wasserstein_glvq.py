"""
Wasserstein GLVQ (WGLVQ).

Generalized LVQ with distributional (Gaussian) prototypes. Each prototype
is a diagonal Gaussian :math:`\\mathcal{N}(\\mu_k, \\text{diag}(\\sigma_k^2))`.
Input samples are treated as Dirac deltas. The classifier distance is the
squared 2-Wasserstein distance:

.. math::

    W_2^2(x, k) = \\sum_j (x_j - \\mu_{kj})^2 + \\sum_j \\sigma_{kj}^2

The variance term acts as a per-prototype "spread penalty" — prototypes with
smaller variance attract nearby points more strongly. The variance is learned
end-to-end via gradient descent on the GLVQ loss.

References
----------
.. [1] Villani, C. (2009). Optimal Transport: Old and New. Springer.
.. [2] Sato, A. & Yamada, K. (1996). Generalized Learning Vector
       Quantization. NIPS.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel, SupervisedState
from prosemble.core.losses import glvq_loss_with_transfer
from prosemble.core.competitions import wtac
from prosemble.core.distance import wasserstein2_distance_matrix


class WGLVQ(SupervisedPrototypeModel):
    """Wasserstein Generalized Learning Vector Quantization.

    Extends GLVQ by replacing point prototypes with Gaussian prototypes.
    Each prototype is parameterized by a mean vector :math:`\\mu_k` and
    log-variance vector :math:`\\log(\\sigma_k^2)`. The squared 2-Wasserstein
    distance between a sample and a prototype combines the squared
    Euclidean distance to the mean with the total prototype variance.

    The variance provides a learnable measure of prototype "uncertainty":

    - Large variance: prototype is uncertain, covers a wider region
    - Small variance: prototype is certain, tightly fits nearby samples

    Parameters
    ----------
    beta : float
        Transfer function steepness parameter. Default: 10.0.
    n_prototypes_per_class : int or dict
        Number of prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    epsilon : float
        Convergence threshold.
    random_seed : int
        Random seed.
    optimizer : str or optax optimizer
        Optimizer name or instance. Default: 'adam'.
    transfer_fn : callable, optional
        Transfer function for loss shaping.
    margin : float
        Margin for loss computation.
    callbacks : list, optional
        Training callbacks.
    use_scan : bool
        Use ``jax.lax.scan`` for training loop.
    batch_size : int, optional
        Mini-batch size.
    lr_scheduler : str or optax.Schedule, optional
        Learning rate schedule.
    lr_scheduler_kwargs : dict, optional
        Scheduler keyword arguments.
    prototypes_initializer : str or callable, optional
        Prototype initialization strategy.
    patience : int, optional
        Early stopping patience.
    restore_best : bool
        Restore best parameters after training.
    class_weight : dict or 'balanced', optional
        Class weights.
    gradient_accumulation_steps : int, optional
        Gradient accumulation steps.
    ema_decay : float, optional
        EMA decay for parameters.
    freeze_params : list of str, optional
        Parameter groups to freeze.
    lookahead : dict, optional
        Lookahead optimizer settings.
    mixed_precision : str, optional
        Mixed precision dtype.

    Attributes
    ----------
    prototype_means_ : array of shape (n_prototypes, n_features)
        Learned prototype mean vectors (after fit).
    prototype_variances_ : array of shape (n_prototypes, n_features)
        Learned prototype variances (after fit).
    """

    def __init__(self, beta=10.0, n_prototypes_per_class=1,
                 max_iter=100, lr=0.01, epsilon=1e-6, random_seed=42,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            n_prototypes_per_class=n_prototypes_per_class,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, distance_fn=None,
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
        self.prototype_means_ = None
        self.prototype_variances_ = None

    def _get_resume_params(self, params):
        return {
            'prototypes': params['prototypes'],
            'log_variances': self._log_variances_,
        }

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )

        # Initialize log-variances from per-class variance
        n_protos = prototypes.shape[0]
        class_vars = []
        for i in range(n_protos):
            mask = y == proto_labels[i]
            class_data = X[mask]
            var = jnp.var(class_data, axis=0)
            class_vars.append(var)
        class_vars = jnp.stack(class_vars)
        log_variances = jnp.log(jnp.maximum(class_vars, 1e-6))

        params = {'prototypes': prototypes, 'log_variances': log_variances}
        opt_state = self._optimizer.init(params)
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        means = params['prototypes']
        log_variances = params['log_variances']
        distances = wasserstein2_distance_matrix(X, means, log_variances)
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _compute_distances_for_rejection(self, X):
        """W2 distances for reject option."""
        log_vars = jnp.log(jnp.maximum(self.prototype_variances_, 1e-6))
        return wasserstein2_distance_matrix(X, self.prototypes_, log_vars)

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.prototype_means_ = params['prototypes']
        self.prototype_variances_ = jnp.exp(params['log_variances'])
        self._log_variances_ = params['log_variances']

    def predict(self, X):
        """Predict class labels using W2 distance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        log_vars = jnp.log(jnp.maximum(self.prototype_variances_, 1e-6))
        distances = wasserstein2_distance_matrix(X, self.prototypes_, log_vars)
        return wtac(distances, self.prototype_labels_)

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.prototype_variances_ is not None:
            attrs.append('prototype_variances_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.prototype_variances_ is not None:
            arrays['prototype_variances_'] = np.asarray(self.prototype_variances_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'prototype_variances_' in arrays:
            self.prototype_variances_ = jnp.asarray(arrays['prototype_variances_'])
            self._log_variances_ = jnp.log(jnp.maximum(self.prototype_variances_, 1e-6))
            self.prototype_means_ = self.prototypes_

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        return hp
