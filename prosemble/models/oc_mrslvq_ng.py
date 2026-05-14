"""
One-Class Matrix RSLVQ with Neural Gas cooperation (OC-MRSLVQ-NG)
and One-Class Local Matrix RSLVQ with NG (OC-LMRSLVQ-NG).

Combines OC-MRSLVQ/OC-LMRSLVQ's Omega metric adaptation and Gaussian
soft-weighting with Neural Gas rank-based neighborhood cooperation:

.. math::

    p(k|x) = \\frac{\\exp(-d_{\\Omega_k} / 2\\sigma^2)}{\\sum_j \\exp(-d_{\\Omega_j} / 2\\sigma^2)}

.. math::

    h_k = \\exp(-\\text{rank}_k / \\gamma)

.. math::

    w_k = \\frac{p(k|x) \\cdot h_k}{\\sum_j p(j|x) \\cdot h_j}

.. math::

    \\text{loss} = \\text{mean}\\left(\\sum_k w_k \\cdot \\text{sigmoid}(\\mu_k + \\text{margin}, \\beta)\\right)

OC-MRSLVQ-NG: global Omega projection matrix (same for all prototypes).
OC-LMRSLVQ-NG: per-prototype :math:`\\Omega_k` matrices (local metric adaptation).

References
----------
.. [1] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization. Neural
       Computation.
.. [2] Seo, S., & Obermayer, K. (2007). Soft Learning Vector
       Quantization. Neural Computation, 19(6):1589-1604.
.. [3] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
"""

import jax.numpy as jnp

from prosemble.models.oc_rslvq_ng_mixin import OCRSLVQNGMixin
from prosemble.models.oc_mrslvq import OCMRSLVQ, OCLMRSLVQ


class OCMRSLVQ_NG(OCRSLVQNGMixin, OCMRSLVQ):
    """One-Class Matrix RSLVQ with Neural Gas neighborhood cooperation.

    Learns a global Omega projection with combined Gaussian + NG
    rank-weighted loss for one-class classification.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture for prototype weighting.
    latent_dim : int, optional
        Dimensionality of the projected space. Default: n_features.
    gamma_init : float, optional
        Initial neighborhood range. Default: n_prototypes / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.
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

    Attributes
    ----------
    omega_ : array of shape (n_features, latent_dim)
        Learned projection matrix.
    thetas_ : array of shape (n_prototypes,)
        Learned per-prototype acceptance thresholds.
    gamma_ : float
        Final gamma value after training.
    """

    def _compute_distances(self, params, X):
        diff = X[:, None, :] - params['prototypes'][None, :, :]
        projected = jnp.einsum('nkd,dl->nkl', diff, params['omega'])
        return jnp.sum(projected ** 2, axis=2)

    def _inference_distances(self, X):
        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('nkd,dl->nkl', diff, self.omega_)
        return jnp.sum(projected ** 2, axis=2)


class OCLMRSLVQ_NG(OCRSLVQNGMixin, OCLMRSLVQ):
    """One-Class Local Matrix RSLVQ with Neural Gas neighborhood cooperation.

    Learns per-prototype :math:`\\Omega_k` projection matrices with combined
    Gaussian + NG rank-weighted loss for one-class classification.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture for prototype weighting.
    latent_dim : int, optional
        Latent space dimensionality per prototype. Default: n_features.
    gamma_init : float, optional
        Initial neighborhood range. Default: n_prototypes / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.
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

    Attributes
    ----------
    omegas_ : array of shape (n_prototypes, n_features, latent_dim)
        Learned per-prototype projection matrices.
    thetas_ : array of shape (n_prototypes,)
        Learned per-prototype acceptance thresholds.
    gamma_ : float
        Final gamma value after training.
    """

    def _compute_distances(self, params, X):
        diff = X[:, None, :] - params['prototypes'][None, :, :]
        projected = jnp.einsum('nkd,kdl->nkl', diff, params['omegas'])
        return jnp.sum(projected ** 2, axis=2)

    def _inference_distances(self, X):
        diff = X[:, None, :] - self.prototypes_[None, :, :]
        projected = jnp.einsum('nkd,kdl->nkl', diff, self.omegas_)
        return jnp.sum(projected ** 2, axis=2)
