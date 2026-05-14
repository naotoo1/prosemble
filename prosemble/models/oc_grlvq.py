"""
One-Class GRLVQ (OC-GRLVQ).

Extends OC-GLVQ with per-feature adaptive relevance weighting.
The distance becomes:

    d_λ(x, w_k) = Σ_j λ_j (x_j - w_{k,j})²

where λ = softmax(relevances) are learned per-feature weights.

References
----------
.. [1] Sato, A., & Yamada, K. (1995). Generalized Learning Vector
       Quantization. NIPS.
.. [2] Hammer, Villmann (2002). Generalized Relevance Learning Vector
       Quantization. Neural Networks, 15(8-9), 1059-1068.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.oc_glvq import OCGLVQ
from prosemble.core.activations import sigmoid_beta


class OCGRLVQ(OCGLVQ):
    """One-Class GRLVQ with per-feature relevance weighting.

    Learns which features are most important for distinguishing
    target from non-target data in a one-class setting.

    Parameters
    ----------
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
    relevances_ : array of shape (n_features,)
        Learned per-feature relevance weights (softmax-normalized).
    """

    def __init__(self, n_prototypes=3, target_label=None, beta=10.0,
                 max_iter=100, lr=0.01, epsilon=1e-6, random_seed=42,
                 distance_fn=None, optimizer='adam', transfer_fn=None,
                 margin=0.0, callbacks=None, use_scan=True,
                 batch_size=None, lr_scheduler=None,
                 lr_scheduler_kwargs=None, prototypes_initializer=None,
                 patience=None, restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
        super().__init__(
            n_prototypes=n_prototypes, target_label=target_label,
            beta=beta, max_iter=max_iter, lr=lr, epsilon=epsilon,
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
        self.relevances_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['relevances'] = self._raw_relevances
        return params

    def _init_state(self, X, y, key):
        state, params, proto_labels = super()._init_state(X, y, key)
        n_features = X.shape[1]
        params['relevances'] = jnp.ones(n_features) / n_features
        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=params['prototypes'],
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        thetas = params['thetas']
        relevances = params['relevances']

        # Relevance-weighted squared Euclidean distances
        lam = jax.nn.softmax(relevances)  # (d,)
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, K, d)
        distances = jnp.sum(lam[None, None, :] * diff ** 2, axis=2)  # (n, K)

        # OC-GLVQ mu
        n = X.shape[0]
        nearest_idx = jnp.argmin(distances, axis=1)
        d_nearest = distances[jnp.arange(n), nearest_idx]
        theta_nearest = thetas[nearest_idx]
        s = jnp.where(y == self._target_label, 1.0, -1.0)
        mu = s * (d_nearest - theta_nearest) / (d_nearest + theta_nearest + 1e-10)
        transfer = self.transfer_fn or sigmoid_beta
        return jnp.mean(transfer(mu + self.margin, self.beta))

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.relevances_ = jax.nn.softmax(params['relevances'])
        self._raw_relevances = params['relevances']

    def decision_function(self, X):
        """Compute scores using relevance-weighted distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        diff = X[:, None, :] - self.prototypes_[None, :, :]
        distances = jnp.sum(
            self.relevances_[None, None, :] * diff ** 2, axis=2
        )
        n = X.shape[0]
        nearest_idx = jnp.argmin(distances, axis=1)
        d_nearest = distances[jnp.arange(n), nearest_idx]
        theta_nearest = self.thetas_[nearest_idx]
        mu = (d_nearest - theta_nearest) / (d_nearest + theta_nearest + 1e-10)
        return 1.0 - jax.nn.sigmoid(self.beta * mu)

    @property
    def relevance_profile(self):
        """Return the learned per-feature relevance weights (normalized)."""
        if self.relevances_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.relevances_

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.relevances_ is not None:
            attrs.append('relevances_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.relevances_ is not None:
            arrays['relevances_'] = np.asarray(self.relevances_)
        if hasattr(self, '_raw_relevances') and self._raw_relevances is not None:
            arrays['_raw_relevances'] = np.asarray(self._raw_relevances)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'relevances_' in arrays:
            self.relevances_ = jnp.asarray(arrays['relevances_'])
        if '_raw_relevances' in arrays:
            self._raw_relevances = jnp.asarray(arrays['_raw_relevances'])
