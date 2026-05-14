"""
Image CBC — Classification-By-Components with a CNN backbone.

Components (classless prototypes) and inputs are both passed through
the same CNN, then detection similarity and reasoning are computed
in the latent space.

Architecture::

    Image (H,W,C) ---> CNN ---> latent_x
                                    |
                                    v
    Component (H,W,C) -> CNN -> latent_c    similarity(latent_x, latent_c)
                                                       |
                                                       v
                                              reasoning -> class_probs
                                                       |
                                                       v
                                                  margin loss

References
----------
.. [1] Saralajew, S., Holdijk, L., & Villmann, T. (2020).
       Classification-by-Components: Probabilistic Modeling of
       Reasoning over a Set of Components. NeurIPS.
"""

import jax
import jax.numpy as jnp

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import cbcc
from prosemble.core.losses import margin_loss
from prosemble.core.similarities import gaussian_similarity
from prosemble.models.lvqmln import _cnn_init, _cnn_forward


class ImageCBC(SupervisedPrototypeModel):
    """Image CBC — CBC with a CNN embedding network.

    Both input images and component images pass through the same CNN
    backbone. Detection similarity and reasoning matrices then produce
    class probabilities.

    Parameters
    ----------
    input_shape : tuple
        Shape of input images as (height, width, channels).
    channels : list of int
        CNN output channels per convolutional layer, e.g. [16, 32].
    kernel_sizes : list of int
        Kernel sizes per convolutional layer, e.g. [3, 3].
    latent_dim : int
        Dimension of the CNN embedding space.
    n_components : int
        Number of components (classless prototypes).
    n_classes : int
        Number of output classes.
    sigma : float
        Bandwidth for Gaussian similarity in component detection.
    activation : str
        Activation function for the CNN backbone. Supported values:
        'relu', 'sigmoid', 'tanh', 'leaky_relu', 'selu'.
    margin : float
        Margin for the margin loss.
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
        E.g. ['backbone'] to freeze the backbone and only train components.
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

    def __init__(self, input_shape=(28, 28, 1), channels=None,
                 kernel_sizes=None, latent_dim=32,
                 n_components=5, n_classes=2, sigma=1.0,
                 activation='relu', margin=0.3,
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None,
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
        self.input_shape = input_shape
        self.channels = channels or [16, 32]
        self.kernel_sizes = kernel_sizes or [3, 3]
        self.latent_dim = latent_dim
        self.n_components = n_components
        self._n_classes = n_classes
        self.sigma = sigma
        self.activation = activation
        self.backbone_params_ = None
        self.components_ = None
        self.reasonings_ = None

    def _get_resume_params(self, params):
        # ImageCBC uses 'components' key instead of 'prototypes'
        return {
            'components': self.components_,
            'reasonings': self.reasonings_,
            'backbone': self.backbone_params_,
        }

    def _init_state(self, X, y, key):
        key1, key2, key3 = jax.random.split(key, 3)

        backbone_params = _cnn_init(
            key1, self.input_shape, self.channels,
            self.kernel_sizes, self.latent_dim, self.activation,
        )

        # Components as images
        X_flat = X.reshape(X.shape[0], -1)
        indices = jax.random.choice(key2, X.shape[0], (self.n_components,), replace=False)
        components = X_flat[indices].reshape(-1, *self.input_shape)

        # Reasoning matrices: (n_components, n_classes, 2)
        reasonings = jnp.ones((self.n_components, self._n_classes, 2)) * 0.5
        reasonings = reasonings + 0.01 * jax.random.normal(key3, reasonings.shape)

        params = {
            'components': components,
            'reasonings': reasonings,
            'backbone': backbone_params,
        }
        opt_state = self._optimizer.init(params)

        # proto_labels not meaningful for CBC
        proto_labels = jnp.zeros(self.n_components, dtype=jnp.int32)

        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=components.reshape(-1, int(jnp.prod(jnp.array(self.input_shape)))),
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        backbone = params['backbone']
        components = params['components']
        reasonings = params['reasonings']

        X_img = X.reshape(-1, *self.input_shape)
        comp_img = components.reshape(-1, *self.input_shape)

        latent_x = _cnn_forward(backbone, X_img, self.activation)
        latent_c = _cnn_forward(backbone, comp_img, self.activation)

        # Similarity in latent space
        diff = latent_x[:, None, :] - latent_c[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=2)
        detections = gaussian_similarity(dist_sq, variance=self.sigma ** 2)

        # CBC reasoning
        class_probs = cbcc(detections, reasonings)

        # Margin loss
        y_one_hot = jax.nn.one_hot(y, self._n_classes)
        return margin_loss(class_probs, y_one_hot, margin=self.margin)

    def _post_update(self, params):
        """Clamp component images to [0, 1] for valid pixel values."""
        if 'components' in params:
            components = jnp.clip(params['components'], 0.0, 1.0)
            return {**params, 'components': components}
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        best_params = kwargs.get('best_params')
        if getattr(self, 'restore_best', False) and best_params is not None:
            params = best_params
        self.prototypes_ = params['components']
        self.components_ = params['components']
        self.reasonings_ = params['reasonings']
        self.backbone_params_ = params['backbone']
        self.prototype_labels_ = proto_labels
        self.loss_ = float(loss_history[-1]) if len(loss_history) > 0 else None
        self.loss_history_ = jnp.array(loss_history)
        self.n_iter_ = n_iter
        val_loss_history = kwargs.get('val_loss_history')
        if val_loss_history is not None:
            self.val_loss_history_ = jnp.array(val_loss_history)
        best_loss = kwargs.get('best_loss')
        if best_loss is not None:
            self.best_loss_ = best_loss

    def predict(self, X):
        proba = self.predict_proba(X)
        return jnp.argmax(proba, axis=1)

    def predict_proba(self, X):
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        X_img = X.reshape(-1, *self.input_shape)
        comp_img = self.components_.reshape(-1, *self.input_shape)

        latent_x = _cnn_forward(self.backbone_params_, X_img, self.activation)
        latent_c = _cnn_forward(self.backbone_params_, comp_img, self.activation)

        diff = latent_x[:, None, :] - latent_c[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=2)
        detections = gaussian_similarity(dist_sq, variance=self.sigma ** 2)
        return cbcc(detections, self.reasonings_)

    def transform(self, X):
        """Transform images to latent space."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _cnn_forward(
            self.backbone_params_, X.reshape(-1, *self.input_shape), self.activation
        )

    def _check_fitted(self):
        if self.components_ is None or self.backbone_params_ is None:
            from prosemble.models.base import NotFittedError
            raise NotFittedError("Model not fitted. Call fit() first.")

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.update({
            'input_shape': list(self.input_shape),
            'channels': self.channels,
            'kernel_sizes': self.kernel_sizes,
            'latent_dim': self.latent_dim,
            'n_components': self.n_components,
            'n_classes': self._n_classes,
            'sigma': self.sigma,
            'activation': self.activation,
        })
        return hp
