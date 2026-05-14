"""
Matrix RSLVQ with Neural Gas Cooperation (MRSLVQ_NG, LMRSLVQ_NG).

Combines RSLVQ's probabilistic soft-assignment with Neural Gas
neighborhood cooperation and learned linear metric adaptation.

MRSLVQ_NG: global Omega matrix + NG cooperation
LMRSLVQ_NG: per-prototype Omega matrices + NG cooperation

When gamma -> 0, recovers standard MRSLVQ / LMRSLVQ behavior.

References
----------
.. [1] Schneider, P., Biehl, M., & Hammer, B. (2009). Adaptive
       Relevance Matrices in Learning Vector Quantization. Neural
       Computation.
.. [2] Seo, S., & Obermayer, K. (2007). Soft Nearest Prototype
       Classification. IEEE Trans. Neural Networks.
.. [3] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init
from prosemble.core.losses import ng_rslvq_loss
from prosemble.core.pooling import stratified_min_pooling


@jit
def _predict_mrslvq_ng_jit(X, prototypes, omega, proto_labels):
    """JIT-compiled MRSLVQ_NG prediction with learned Omega metric."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,dl->npl', diff, omega)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


@partial(jit, static_argnums=(4,))
def _predict_proba_mrslvq_ng_jit(X, prototypes, omega, proto_labels, n_classes):
    """JIT-compiled MRSLVQ_NG probability prediction."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,dl->npl', diff, omega)
    distances = jnp.sum(projected ** 2, axis=2)
    class_dists = stratified_min_pooling(distances, proto_labels, n_classes)
    return jax.nn.softmax(-class_dists, axis=1)


@jit
def _predict_lmrslvq_ng_jit(X, prototypes, omegas, proto_labels):
    """JIT-compiled LMRSLVQ_NG prediction with per-prototype Omega metrics."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,pdl->npl', diff, omegas)
    distances = jnp.sum(projected ** 2, axis=2)
    return wtac(distances, proto_labels)


@partial(jit, static_argnums=(4,))
def _predict_proba_lmrslvq_ng_jit(X, prototypes, omegas, proto_labels, n_classes):
    """JIT-compiled LMRSLVQ_NG probability prediction."""
    diff = X[:, None, :] - prototypes[None, :, :]
    projected = jnp.einsum('npd,pdl->npl', diff, omegas)
    distances = jnp.sum(projected ** 2, axis=2)
    class_dists = stratified_min_pooling(distances, proto_labels, n_classes)
    return jax.nn.softmax(-class_dists, axis=1)


class MRSLVQ_NG(SupervisedPrototypeModel):
    """Matrix Robust Soft LVQ with Neural Gas Cooperation.

    Combines:

    - RSLVQ probabilistic loss: ``-log(P(correct|x))``
    - Neural Gas cooperation: all prototypes weighted by rank via
      ``exp(-rank / gamma)``
    - Global Omega matrix for metric adaptation:
      ``d(x, w) = (x - w)^T Omega^T Omega (x - w)``

    Parameters
    ----------
    sigma : float
        Bandwidth for RSLVQ Gaussian mixture probability computation.
    latent_dim : int, optional
        Dimensionality of the Omega projection space. If None, uses input dim.
    gamma_init : float, optional
        Initial neighborhood range for NG cooperation.
        Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay factor for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
    rejection_confidence : float, optional
        Minimum class probability for confident prediction (0 to 1).
        Samples below this threshold are rejected (label -1).
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

    def __init__(self, sigma=1.0, latent_dim=None, gamma_init=None,
                 gamma_final=0.01, gamma_decay=None,
                 rejection_confidence=None, n_prototypes_per_class=1,
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
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.rejection_confidence = rejection_confidence
        self.omega_ = None
        self.gamma_ = None

        # Freeze gamma from optimizer
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        params['omega'] = self.omega_
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        omega = identity_omega_init(n_features, latent_dim)

        # Compute gamma_init from prototype count
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)

        gamma_init = (self.gamma_init if self.gamma_init is not None
                      else max_per_class / 2.0)
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (
                self.gamma_final / gamma_init
            ) ** (1.0 / self.max_iter)

        params = {
            'prototypes': prototypes,
            'omega': omega,
            'gamma': jnp.array(gamma_init, dtype=jnp.float32),
        }
        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        omega = params['omega']
        gamma = params['gamma']
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,dl->npl', diff, omega)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)
        return ng_rslvq_loss(distances, y, proto_labels,
                             sigma=self.sigma, gamma=gamma)

    def _post_update(self, params):
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omega_ = params['omega']
        self.gamma_ = float(params['gamma'])

    @property
    def omega_matrix(self):
        """Return the learned Omega matrix."""
        if self.omega_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_

    @property
    def lambda_matrix(self):
        """Return Lambda = Omega^T Omega (relevance matrix)."""
        if self.omega_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_.T @ self.omega_

    def predict(self, X):
        """Predict using learned Omega distance."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_mrslvq_ng_jit(
            X, self.prototypes_, self.omega_, self.prototype_labels_
        )

    def predict_proba(self, X):
        """Predict class probabilities using Omega-projected distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_proba_mrslvq_ng_jit(
            X, self.prototypes_, self.omega_, self.prototype_labels_,
            self.n_classes_
        )

    def predict_with_rejection(self, X, confidence=None):
        """Predict with rejection option.

        Samples whose maximum class probability is below the confidence
        threshold are assigned label -1 (rejected).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        confidence : float, optional
            Override the model's rejection_confidence for this call.

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        self._check_fitted()
        threshold = (confidence if confidence is not None
                     else self.rejection_confidence)
        if threshold is None:
            return self.predict(X)

        X = jnp.asarray(X, dtype=jnp.float32)
        proba = self.predict_proba(X)
        max_proba = jnp.max(proba, axis=1)
        preds = jnp.argmax(proba, axis=1)
        return jnp.where(max_proba >= threshold, preds, -1)

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omega_ is not None:
            attrs.append('omega_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omega_ is not None:
            arrays['omega_'] = np.asarray(self.omega_)
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omega_' in arrays:
            self.omega_ = jnp.asarray(arrays['omega_'])
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma'] = self.sigma
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        hp['rejection_confidence'] = self.rejection_confidence
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp


class LMRSLVQ_NG(SupervisedPrototypeModel):
    """Localized Matrix Robust Soft LVQ with Neural Gas Cooperation.

    Each prototype k has its own Omega_k matrix. Combined with RSLVQ
    probabilistic loss and NG rank-based neighborhood cooperation.

    Parameters
    ----------
    sigma : float
        Bandwidth for RSLVQ Gaussian mixture probability computation.
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
    rejection_confidence : float, optional
        Minimum class probability for confident prediction (0 to 1).
        Samples below this threshold are rejected (label -1).
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

    def __init__(self, sigma=1.0, latent_dim=None, gamma_init=None,
                 gamma_final=0.01, gamma_decay=None,
                 rejection_confidence=None, n_prototypes_per_class=1,
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
        self.sigma = sigma
        self.latent_dim = latent_dim
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.rejection_confidence = rejection_confidence
        self.omegas_ = None
        self.gamma_ = None

        # Freeze gamma from optimizer
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        params['omegas'] = self.omegas_
        gamma = self.gamma_ if self.gamma_ is not None else (
            self._gamma_init_actual if hasattr(self, '_gamma_init_actual') else 1.0
        )
        params['gamma'] = jnp.array(gamma, dtype=jnp.float32)
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        n_protos = prototypes.shape[0]
        omega_single = identity_omega_init(n_features, latent_dim)
        omegas = jnp.tile(omega_single[None, :, :], (n_protos, 1, 1))

        # Compute gamma_init from prototype count
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)

        gamma_init = (self.gamma_init if self.gamma_init is not None
                      else max_per_class / 2.0)
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (
                self.gamma_final / gamma_init
            ) ** (1.0 / self.max_iter)

        params = {
            'prototypes': prototypes,
            'omegas': omegas,
            'gamma': jnp.array(gamma_init, dtype=jnp.float32),
        }
        opt_state = self._optimizer.init(params)
        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        prototypes = params['prototypes']
        omegas = params['omegas']  # (p, d, l)
        gamma = params['gamma']
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        projected = jnp.einsum('npd,pdl->npl', diff, omegas)  # (n, p, l)
        distances = jnp.sum(projected ** 2, axis=2)  # (n, p)
        return ng_rslvq_loss(distances, y, proto_labels,
                             sigma=self.sigma, gamma=gamma)

    def _post_update(self, params):
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omegas_ = params['omegas']
        self.gamma_ = float(params['gamma'])

    def predict(self, X):
        """Predict using local Omega distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_lmrslvq_ng_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_
        )

    def predict_proba(self, X):
        """Predict class probabilities using local Omega-projected distances."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_proba_lmrslvq_ng_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_,
            self.n_classes_
        )

    def predict_with_rejection(self, X, confidence=None):
        """Predict with rejection option.

        Samples whose maximum class probability is below the confidence
        threshold are assigned label -1 (rejected).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        confidence : float, optional
            Override the model's rejection_confidence for this call.

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        self._check_fitted()
        threshold = (confidence if confidence is not None
                     else self.rejection_confidence)
        if threshold is None:
            return self.predict(X)

        X = jnp.asarray(X, dtype=jnp.float32)
        proba = self.predict_proba(X)
        max_proba = jnp.max(proba, axis=1)
        preds = jnp.argmax(proba, axis=1)
        return jnp.where(max_proba >= threshold, preds, -1)

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omegas_ is not None:
            attrs.append('omegas_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omegas_ is not None:
            arrays['omegas_'] = np.asarray(self.omegas_)
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omegas_' in arrays:
            self.omegas_ = jnp.asarray(arrays['omegas_'])
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma'] = self.sigma
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        hp['rejection_confidence'] = self.rejection_confidence
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
