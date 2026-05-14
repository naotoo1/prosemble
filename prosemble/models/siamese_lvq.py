"""
Siamese LVQ models: SiameseGLVQ, SiameseGMLVQ, SiameseGTLVQ.

In Siamese variants, an MLP backbone transforms BOTH inputs AND
prototypes before computing distances. Prototypes remain in the
original input space and are projected at each step.

This contrasts with LVQMLN, where only inputs are transformed and
prototypes live directly in latent space.

Architecture::

    Input (d) ---> backbone ---> latent_x
                                    |
                                    v
    Prototype (d) -> backbone -> latent_w     distance(latent_x, latent_w)
                                                        |
                                                        v
                                                    LVQ loss

References
----------
.. [1] Villmann, T., et al. (2017). Prototype-based Neural Network
       Layers: Incorporating Vector Quantization. arXiv:1812.01214.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.losses import glvq_loss_with_transfer
from prosemble.core.distance import squared_euclidean_distance_matrix
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init, random_omega_init
from prosemble.models.lvqmln import _mlp_init, _mlp_forward
from prosemble.core.utils import orthogonalize


class SiameseGLVQ(SupervisedPrototypeModel):
    """Siamese GLVQ — GLVQ with a learned embedding network.

    Both inputs and prototypes are transformed through the same MLP
    backbone before computing squared Euclidean distances.

    Parameters
    ----------
    hidden_sizes : list of int
        Hidden layer sizes for the backbone MLP.
    latent_dim : int
        Dimension of the embedding space.
    activation : str
        Activation function for the backbone MLP. Supported values:
        'sigmoid', 'relu', 'tanh', 'leaky_relu', 'selu'.
    beta : float
        Transfer function parameter for GLVQ loss.
    bb_lr : float, optional
        Separate learning rate for the backbone network. If None,
        uses the same lr as prototypes. Default: None.
    both_path_gradients : bool
        If True, compute gradients through both input and prototype
        paths. If False, prototype path gradients are stopped.
        Default: True.
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
        - 'sync_period': int (default 6) — sync every k steps
        - 'slow_step_size': float (default 0.5) — interpolation factor
        Default: None (no lookahead).
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training. 'float16' or 'bfloat16'.
        Master weights stay in float32; forward/backward pass runs in lower
        precision for ~2x speed and ~half memory on GPU. Float16 uses static
        loss scaling to prevent gradient underflow. Default: None (disabled).
    """

    def __init__(self, hidden_sizes=None, latent_dim=2,
                 activation='sigmoid', beta=10.0, bb_lr=None,
                 both_path_gradients=True,
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
        self.hidden_sizes = hidden_sizes or [10]
        self.latent_dim = latent_dim
        self.activation = activation
        self.beta = beta
        self.bb_lr = bb_lr
        self.both_path_gradients = both_path_gradients
        self.backbone_params_ = None

        if bb_lr is not None:
            self._optimizer = self._build_multi_lr_optimizer(
                self._optimizer_spec, self.lr, bb_lr
            )

    def _build_multi_lr_optimizer(self, optimizer, proto_lr, bb_lr):
        """Build optimizer with separate learning rates for prototypes and backbone."""
        import optax
        if not isinstance(optimizer, str):
            return optimizer
        proto_opt = self._build_optimizer(optimizer, proto_lr)
        bb_opt = self._build_optimizer(optimizer, bb_lr)
        return optax.multi_transform(
            {'prototypes': proto_opt, 'backbone': bb_opt},
            param_labels=lambda params: {k: k for k in params},
        )

    def _get_resume_params(self, params):
        params['backbone'] = self.backbone_params_
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        layer_sizes = [n_features] + list(self.hidden_sizes) + [self.latent_dim]

        key1, key2 = jax.random.split(key)
        backbone_params = _mlp_init(key1, layer_sizes, self.activation)

        # Prototypes in input space (they get projected through backbone)
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key2
        )

        params = {
            'prototypes': prototypes,
            'backbone': backbone_params,
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
        backbone = params['backbone']
        latent_x = _mlp_forward(backbone, X, self.activation)
        latent_w = _mlp_forward(backbone, params['prototypes'], self.activation)
        if not self.both_path_gradients:
            latent_w = jax.lax.stop_gradient(latent_w)
        distances = squared_euclidean_distance_matrix(latent_x, latent_w)
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.backbone_params_ = params['backbone']

    def predict(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        latent_x = _mlp_forward(self.backbone_params_, X, self.activation)
        latent_w = _mlp_forward(self.backbone_params_, self.prototypes_, self.activation)
        distances = squared_euclidean_distance_matrix(latent_x, latent_w)
        return wtac(distances, self.prototype_labels_)

    def predict_proba(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        latent_x = _mlp_forward(self.backbone_params_, X, self.activation)
        latent_w = _mlp_forward(self.backbone_params_, self.prototypes_, self.activation)
        distances = squared_euclidean_distance_matrix(latent_x, latent_w)
        from prosemble.core.pooling import stratified_min_pooling
        class_dists = stratified_min_pooling(
            distances, self.prototype_labels_, self.n_classes_
        )
        return jax.nn.softmax(-class_dists, axis=1)

    def transform(self, X):
        """Transform data through the backbone."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _mlp_forward(self.backbone_params_, X, self.activation)

    def _check_fitted(self):
        if self.prototypes_ is None or self.backbone_params_ is None:
            from prosemble.models.base import NotFittedError
            raise NotFittedError("Model not fitted. Call fit() first.")

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'hidden_sizes': self.hidden_sizes,
            'latent_dim': self.latent_dim,
            'activation': self.activation,
            'beta': self.beta,
        })
        return hp


class SiameseGMLVQ(SupervisedPrototypeModel):
    """Siamese GMLVQ — GMLVQ with a learned embedding network.

    Both inputs and prototypes are transformed through the same MLP,
    then distances are computed using a learned :math:`\\Omega` matrix in the
    latent space:

    .. math::

        d = \\|\\Omega(f(x) - f(w))\\|^2

    Parameters
    ----------
    hidden_sizes : list of int
        Hidden layer sizes for the backbone MLP.
    latent_dim : int
        Dimension of the backbone output (embedding space).
    omega_dim : int, optional
        Omega mapping dimension (number of rows in Omega). If None,
        uses latent_dim (square matrix). Default: None.
    activation : str
        Activation function for the backbone MLP. Supported values:
        'sigmoid', 'relu', 'tanh', 'leaky_relu', 'selu'.
    beta : float
        Transfer function parameter for GLVQ loss.
    bb_lr : float, optional
        Separate learning rate for the backbone network. If None,
        uses the same lr as prototypes. Default: None.
    both_path_gradients : bool
        If True, compute gradients through both input and prototype
        paths. If False, prototype path gradients are stopped.
        Default: True.
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
        - 'sync_period': int (default 6) — sync every k steps
        - 'slow_step_size': float (default 0.5) — interpolation factor
        Default: None (no lookahead).
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training. 'float16' or 'bfloat16'.
        Master weights stay in float32; forward/backward pass runs in lower
        precision for ~2x speed and ~half memory on GPU. Float16 uses static
        loss scaling to prevent gradient underflow. Default: None (disabled).
    """

    def __init__(self, hidden_sizes=None, latent_dim=2,
                 omega_dim=None, activation='sigmoid', beta=10.0,
                 bb_lr=None, both_path_gradients=True,
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
        self.hidden_sizes = hidden_sizes or [10]
        self.latent_dim = latent_dim
        self.omega_dim = omega_dim
        self.activation = activation
        self.beta = beta
        self.bb_lr = bb_lr
        self.both_path_gradients = both_path_gradients
        self.backbone_params_ = None
        self.omega_ = None

        if bb_lr is not None:
            self._optimizer = self._build_multi_lr_optimizer(
                self._optimizer_spec, self.lr, bb_lr
            )

    def _build_multi_lr_optimizer(self, optimizer, proto_lr, bb_lr):
        """Build optimizer with separate learning rates for prototypes and backbone."""
        import optax
        if not isinstance(optimizer, str):
            return optimizer
        proto_opt = self._build_optimizer(optimizer, proto_lr)
        bb_opt = self._build_optimizer(optimizer, bb_lr)
        return optax.multi_transform(
            {'prototypes': proto_opt, 'backbone': bb_opt, 'omega': proto_opt},
            param_labels=lambda params: {k: k for k in params},
        )

    def _get_resume_params(self, params):
        params['backbone'] = self.backbone_params_
        params['omega'] = self.omega_
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        layer_sizes = [n_features] + list(self.hidden_sizes) + [self.latent_dim]
        omega_dim = self.omega_dim or self.latent_dim

        key1, key2 = jax.random.split(key)
        backbone_params = _mlp_init(key1, layer_sizes, self.activation)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key2
        )
        omega = identity_omega_init(self.latent_dim, omega_dim)

        params = {
            'prototypes': prototypes,
            'backbone': backbone_params,
            'omega': omega,
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
        backbone = params['backbone']
        omega = params['omega']
        latent_x = _mlp_forward(backbone, X, self.activation)
        latent_w = _mlp_forward(backbone, params['prototypes'], self.activation)
        if not self.both_path_gradients:
            latent_w = jax.lax.stop_gradient(latent_w)
        # Omega distance in latent space
        diff = latent_x[:, None, :] - latent_w[None, :, :]
        projected = jnp.einsum('npd,dl->npl', diff, omega)
        distances = jnp.sum(projected ** 2, axis=2)
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.backbone_params_ = params['backbone']
        self.omega_ = params['omega']

    def predict(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        latent_x = _mlp_forward(self.backbone_params_, X, self.activation)
        latent_w = _mlp_forward(self.backbone_params_, self.prototypes_, self.activation)
        diff = latent_x[:, None, :] - latent_w[None, :, :]
        projected = jnp.einsum('npd,dl->npl', diff, self.omega_)
        distances = jnp.sum(projected ** 2, axis=2)
        return wtac(distances, self.prototype_labels_)

    def transform(self, X):
        """Transform data through the backbone."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _mlp_forward(self.backbone_params_, X, self.activation)

    @property
    def lambda_matrix(self):
        """Return Lambda = Omega^T Omega in latent space."""
        if self.omega_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_.T @ self.omega_

    def _check_fitted(self):
        if self.prototypes_ is None or self.backbone_params_ is None:
            from prosemble.models.base import NotFittedError
            raise NotFittedError("Model not fitted. Call fit() first.")

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'hidden_sizes': self.hidden_sizes,
            'latent_dim': self.latent_dim,
            'activation': self.activation,
            'beta': self.beta,
        })
        if self.omega_dim is not None:
            hp['omega_dim'] = self.omega_dim
        return hp


class SiameseGTLVQ(SupervisedPrototypeModel):
    """Siamese GTLVQ — GTLVQ with a learned embedding network.

    Both inputs and prototypes are transformed through the same MLP,
    then tangent distances are computed in the latent space using
    per-prototype subspace bases.

    Parameters
    ----------
    hidden_sizes : list of int
        Hidden layer sizes for the backbone MLP.
    latent_dim : int
        Dimension of the backbone output (embedding space).
    subspace_dim : int
        Tangent subspace dimension per prototype. Each prototype gets a
        learned orthonormal basis of this rank in latent space.
    activation : str
        Activation function for the backbone MLP. Supported values:
        'sigmoid', 'relu', 'tanh', 'leaky_relu', 'selu'.
    beta : float
        Transfer function parameter for GLVQ loss.
    bb_lr : float, optional
        Separate learning rate for the backbone network. If None,
        uses the same lr as prototypes. Default: None.
    both_path_gradients : bool
        If True, compute gradients through both input and prototype
        paths. If False, prototype path gradients are stopped.
        Default: True.
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
        - 'sync_period': int (default 6) — sync every k steps
        - 'slow_step_size': float (default 0.5) — interpolation factor
        Default: None (no lookahead).
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training. 'float16' or 'bfloat16'.
        Master weights stay in float32; forward/backward pass runs in lower
        precision for ~2x speed and ~half memory on GPU. Float16 uses static
        loss scaling to prevent gradient underflow. Default: None (disabled).
    """

    def __init__(self, hidden_sizes=None, latent_dim=4,
                 subspace_dim=2, activation='sigmoid', beta=10.0,
                 bb_lr=None, both_path_gradients=True,
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
        self.hidden_sizes = hidden_sizes or [10]
        self.latent_dim = latent_dim
        self.subspace_dim = subspace_dim
        self.activation = activation
        self.beta = beta
        self.bb_lr = bb_lr
        self.both_path_gradients = both_path_gradients
        self.backbone_params_ = None
        self.omegas_ = None

        if bb_lr is not None:
            self._optimizer = self._build_multi_lr_optimizer(
                self._optimizer_spec, self.lr, bb_lr
            )

    def _build_multi_lr_optimizer(self, optimizer, proto_lr, bb_lr):
        """Build optimizer with separate learning rates for prototypes and backbone."""
        import optax
        if not isinstance(optimizer, str):
            return optimizer
        proto_opt = self._build_optimizer(optimizer, proto_lr)
        bb_opt = self._build_optimizer(optimizer, bb_lr)
        return optax.multi_transform(
            {'prototypes': proto_opt, 'backbone': bb_opt, 'omegas': proto_opt},
            param_labels=lambda params: {k: k for k in params},
        )

    def _get_resume_params(self, params):
        params['backbone'] = self.backbone_params_
        params['omegas'] = self.omegas_
        return params

    def _init_state(self, X, y, key):
        n_features = X.shape[1]
        layer_sizes = [n_features] + list(self.hidden_sizes) + [self.latent_dim]

        key1, key2, key3 = jax.random.split(key, 3)
        backbone_params = _mlp_init(key1, layer_sizes, self.activation)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key2
        )
        n_protos = prototypes.shape[0]

        # Per-prototype tangent bases in latent space
        keys = jax.random.split(key3, n_protos)
        omegas = jnp.stack([
            random_omega_init(self.latent_dim, self.subspace_dim, k) for k in keys
        ])  # (p, latent_dim, subspace_dim)

        params = {
            'prototypes': prototypes,
            'backbone': backbone_params,
            'omegas': omegas,
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
        backbone = params['backbone']
        omegas = params['omegas']  # (p, latent_dim, subspace_dim)
        latent_x = _mlp_forward(backbone, X, self.activation)
        latent_w = _mlp_forward(backbone, params['prototypes'], self.activation)
        if not self.both_path_gradients:
            latent_w = jax.lax.stop_gradient(latent_w)
        # Tangent distance in latent space
        diff = latent_x[:, None, :] - latent_w[None, :, :]  # (n, p, latent_dim)
        proj = jnp.einsum('npd,pds->nps', diff, omegas)  # (n, p, s)
        recon = jnp.einsum('nps,pds->npd', proj, omegas)  # (n, p, latent_dim)
        tang_diff = diff - recon
        distances = jnp.sum(tang_diff ** 2, axis=2)  # (n, p)
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _post_update(self, params):
        """Re-orthogonalize tangent bases."""
        if 'omegas' not in params:
            return params
        omegas = jax.vmap(orthogonalize)(params['omegas'])
        return {**params, 'omegas': omegas}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.backbone_params_ = params['backbone']
        self.omegas_ = params['omegas']

    def predict(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        latent_x = _mlp_forward(self.backbone_params_, X, self.activation)
        latent_w = _mlp_forward(self.backbone_params_, self.prototypes_, self.activation)
        diff = latent_x[:, None, :] - latent_w[None, :, :]
        proj = jnp.einsum('npd,pds->nps', diff, self.omegas_)
        recon = jnp.einsum('nps,pds->npd', proj, self.omegas_)
        tang_diff = diff - recon
        distances = jnp.sum(tang_diff ** 2, axis=2)
        return wtac(distances, self.prototype_labels_)

    def transform(self, X):
        """Transform data through the backbone."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _mlp_forward(self.backbone_params_, X, self.activation)

    def _check_fitted(self):
        if self.prototypes_ is None or self.backbone_params_ is None:
            from prosemble.models.base import NotFittedError
            raise NotFittedError("Model not fitted. Call fit() first.")

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'hidden_sizes': self.hidden_sizes,
            'latent_dim': self.latent_dim,
            'subspace_dim': self.subspace_dim,
            'activation': self.activation,
            'beta': self.beta,
        })
        return hp
