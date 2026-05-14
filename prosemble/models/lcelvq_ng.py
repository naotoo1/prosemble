"""
Localized Matrix Cross-Entropy LVQ with Neural Gas cooperation (LCELVQ-NG).

Combines CELVQ-NG's cross-entropy loss over NG-weighted softmax logits
with per-prototype :math:`\\Omega_k` matrices that learn local metrics adapted to
each prototype's region:

.. math::

    d(x, w_k) = \\|\\Omega_k(x - w_k)\\|^2

Each prototype learns its own discriminative subspace while Neural Gas
rank-based cooperation ensures robust prototype placement. The cross-
entropy loss over all classes simultaneously provides gradient flow to
all local matrices.

When :math:`\\gamma \\to 0`, only the nearest prototype per class dominates and
LCELVQ-NG recovers a localized matrix variant of standard CELVQ.

References
----------
.. [1] Villmann, T., et al. (2019). Analysis of variants of
       classification learning vector quantization by a stochastic
       setting.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
.. [3] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from prosemble.models.celvq_ng_mixin import CELVQNGMixin
from prosemble.models.crossentropy_lvq import CELVQ
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init


@jit
def _predict_lcelvq_ng_jit(X, prototypes, omegas, proto_labels):
    """JIT-compiled LCELVQ-NG prediction with per-prototype Omega metrics."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,pdl->npl', diff, omegas)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


class LCELVQ_NG(CELVQNGMixin, CELVQ):
    """Localized Matrix Cross-Entropy LVQ with Neural Gas cooperation.

    Combines three key ideas:

    - Cross-entropy loss: softmax over all-class NG-weighted distances
    - Neural Gas cooperation: all same-class prototypes participate,
      weighted by rank via :math:`\\exp(-\\text{rank} / \\gamma)`
    - Per-prototype :math:`\\Omega_k`:

      .. math::

          d(x, w_k) = \\|\\Omega_k(x - w_k)\\|^2

      learns local metrics adapted to each prototype's region

    The neighborhood range :math:`\\gamma` decays during training from
    :math:`\\gamma_{\\text{init}}` to :math:`\\gamma_{\\text{final}}`.
    When :math:`\\gamma \\to 0`, LCELVQ-NG recovers a localized
    matrix CELVQ.

    Parameters
    ----------
    latent_dim : int, optional
        Latent space dimensionality per prototype. If None, uses input dim.
    gamma_init : float, optional
        Initial neighborhood range for NG cooperation.
        Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay factor for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
    n_prototypes_per_class : int
        Number of prototypes per class.
    max_iter : int
        Maximum training iterations.
    lr : float
        Learning rate.
    epsilon : float
        Convergence threshold on loss change.
    random_seed : int
        Random seed for reproducibility.
    distance_fn : callable, optional
        Distance function (default: squared Euclidean).
    optimizer : str or optax optimizer, optional
        Optimizer name ('adam', 'sgd') or optax GradientTransformation.
        Default: 'adam'.
    transfer_fn : callable, optional
        Transfer function for loss shaping (default: identity).
    margin : float
        Margin for loss computation.
    callbacks : list, optional
        List of Callback objects.
    use_scan : bool
        If True (default), use jax.lax.scan for training (faster, JIT-compiled,
        but runs all max_iter iterations even after convergence).
        If False, use a Python for-loop with true early stopping (no wasted
        compute after convergence, but slower per iteration).
    batch_size : int, optional
        Mini-batch size. If None (default), use full-batch training.
        When set, each epoch iterates over shuffled mini-batches of this size.
    lr_scheduler : str or optax.Schedule, optional
        Learning rate schedule. Supported strings: 'exponential_decay',
        'cosine_decay', 'warmup_cosine_decay', 'warmup_exponential_decay',
        'warmup_constant', 'polynomial', 'linear', 'piecewise_constant',
        'sgdr'. Or pass a custom optax.Schedule. Default: None.
    lr_scheduler_kwargs : dict, optional
        Keyword arguments passed to the learning rate scheduler
        (e.g. ``decay_rate``, ``transition_steps``). Default: None.
    prototypes_initializer : str or callable, optional
        How to initialize prototypes. Supported strings: 'stratified_random'
        (default), 'class_mean', 'class_conditional_mean', 'stratified_noise',
        'random_normal', 'uniform', 'zeros', 'ones', 'fill_value'.
        Or pass a callable ``(X, y, n_per_class, key) -> (protos, labels)``.
    patience : int, optional
        Number of consecutive epochs with no improvement before stopping.
        If None (default), stops after a single non-improving step (epsilon
        check). Requires use_scan=False for true early stopping.
    restore_best : bool
        If True, restore the parameters that achieved the lowest loss
        (or validation loss if validation data is provided). Default: False.
    class_weight : dict or 'balanced', optional
        Weights for each class. Dict maps class label to weight, e.g.
        {0: 1.0, 1: 2.0, 2: 1.5}. 'balanced' auto-computes weights
        inversely proportional to class frequencies. Default: None (uniform).
    gradient_accumulation_steps : int, optional
        Accumulate gradients over this many steps before applying an update.
        Effective batch size = batch_size * gradient_accumulation_steps.
        Default: None (no accumulation).
    ema_decay : float, optional
        Exponential moving average decay for parameters (0 < ema_decay < 1).
        After training, model parameters are replaced with EMA-smoothed values.
        Typical values: 0.999, 0.9999. Default: None (no EMA).
    freeze_params : list of str, optional
        List of parameter group names to freeze (zero gradients).
        E.g. ['backbone'] to freeze the backbone and only train prototypes.
        Default: None (all parameters trainable).
    lookahead : dict, optional
        Enable lookahead optimizer wrapper. Dict with keys:
        - 'sync_period': int (default 6) -- sync every k steps
        - 'slow_step_size': float (default 0.5) -- interpolation factor
        Default: None (no lookahead).
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training. 'float16' or 'bfloat16'.
        Master weights stay in float32; forward/backward pass runs in lower
        precision for ~2x speed and ~half memory on GPU. Float16 uses static
        loss scaling to prevent gradient underflow. Default: None (disabled).
    """

    def __init__(self, latent_dim=None, gamma_init=None,
                 gamma_final=0.01, gamma_decay=None,
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            gamma_init=gamma_init, gamma_final=gamma_final,
            gamma_decay=gamma_decay,
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
        self.latent_dim = latent_dim
        self.omegas_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['omegas'] = self.omegas_
        return params

    def _init_metric_params(self, params, X, prototypes, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        n_protos = prototypes.shape[0]
        omega_single = identity_omega_init(n_features, latent_dim)
        omegas = jnp.tile(omega_single[None, :, :], (n_protos, 1, 1))
        params['omegas'] = omegas
        return params

    def _compute_distances(self, params, X):
        diff = X[:, None, :] - params['prototypes'][None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,pdl->npl', diff, params['omegas'])  # (n, p, l)
        return jnp.sum(projected ** 2, axis=2)  # (n, p)

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omegas_ = params['omegas']

    def predict(self, X):
        """Predict using per-prototype :math:`\\Omega_k` distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_lcelvq_ng_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_
        )

    def predict_proba(self, X):
        """Predict calibrated probabilities using per-prototype :math:`\\Omega_k` distances.

        Uses per-class min pooling with the learned local :math:`\\Omega_k` metrics,
        consistent with the training objective.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        # Compute per-prototype Omega-transformed distances
        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('npd,pdl->npl', diff, self.omegas_)
        distances = jnp.sum(projected ** 2, axis=2)

        # Softmax over per-class min distances
        from prosemble.core.pooling import stratified_min_pooling
        class_dists = stratified_min_pooling(
            distances, self.prototype_labels_, self.n_classes_
        )
        return jax.nn.softmax(-class_dists, axis=1)

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omegas_ is not None:
            attrs.append('omegas_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omegas_ is not None:
            arrays['omegas_'] = np.asarray(self.omegas_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omegas_' in arrays:
            self.omegas_ = jnp.asarray(arrays['omegas_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
