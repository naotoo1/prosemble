"""
Classification-By-Components (CBC).

A reasoning-based classification model where components (prototypes
without class labels) learn both detection (similarity) and reasoning
(positive/negative evidence per class).

References
----------
.. [1] Saralajew, S., Holdijk, L., & Villmann, T. (2020).
       Classification-by-Components: Probabilistic Modeling of
       Reasoning over a Set of Components. NeurIPS.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from prosemble.models.prototype_base import SupervisedPrototypeModel, NotFittedError
from prosemble.core.competitions import cbcc
from prosemble.core.losses import margin_loss
from prosemble.core.similarities import gaussian_similarity


@jit
def _predict_proba_cbc_jit(X, components, reasonings, sigma_sq):
    """JIT-compiled CBC probability prediction."""
    diff = X[:, None, :] - components[None, :, :]
    dist_sq = jnp.sum(diff ** 2, axis=2)
    detections = gaussian_similarity(dist_sq, variance=sigma_sq)
    return cbcc(detections, reasonings)


@jit
def _predict_cbc_jit(X, components, reasonings, sigma_sq):
    """JIT-compiled CBC class prediction."""
    probs = _predict_proba_cbc_jit(X, components, reasonings, sigma_sq)
    return jnp.argmax(probs, axis=1)


class CBC(SupervisedPrototypeModel):
    """Classification-By-Components.

    Components detect patterns in the input (via similarity), then
    reasoning matrices determine how each detection contributes
    evidence for/against each class.

    Parameters
    ----------
    n_components : int
        Number of components (analogous to prototypes, but classless).
    n_classes : int
        Number of output classes.
    sigma : float
        Bandwidth for Gaussian similarity in component detection.
    margin : float
        Margin for the margin loss.
    components_initializer : callable, optional
        Initializer for component vectors. Signature:
        ``(X, key, n_components) -> components``. Default: None
        (selects random training samples).
    reasonings_initializer : callable, optional
        Initializer for the reasoning matrix. Signature:
        ``(n_components, n_classes, key) -> reasonings``. Default: None
        (initializes near-uniform with small noise).
    similarity_fn : callable, optional
        Similarity function for component detection. Default: None
        (uses Gaussian similarity).
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
        E.g. ['components'] to freeze the components and only train reasonings.
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

    def __init__(self, n_components=5, n_classes=2, sigma=1.0,
                 margin=0.3, components_initializer=None,
                 reasonings_initializer=None, similarity_fn=None,
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
        self.n_components = n_components
        self._n_classes = n_classes
        self.sigma = sigma
        self.components_initializer = components_initializer
        self.reasonings_initializer = reasonings_initializer
        if similarity_fn is not None:
            self._similarity_fn = similarity_fn
        else:
            self._similarity_fn = gaussian_similarity
        self.components_ = None
        self.reasonings_ = None

    def _get_resume_params(self, params):
        # CBC uses 'components' key instead of 'prototypes'
        return {'components': self.components_, 'reasonings': self.reasonings_}

    def _init_state(self, X, y, key):
        n_classes = self._n_classes
        key1, key2 = jax.random.split(key)

        # Initialize components
        if self.components_initializer is not None:
            components = self.components_initializer(X, key1, self.n_components)
        else:
            indices = jax.random.choice(key1, X.shape[0], (self.n_components,), replace=False)
            components = X[indices]

        # Initialize reasoning matrices: (n_components, n_classes, 2)
        if self.reasonings_initializer is not None:
            reasonings = self.reasonings_initializer(
                self.n_components, n_classes, key2
            )
        else:
            # Default: near uniform p=0.5, n=0.5 with small noise
            reasonings = jnp.ones((self.n_components, n_classes, 2)) * 0.5
            reasonings = reasonings + 0.01 * jax.random.normal(key2, reasonings.shape)

        params = {'components': components, 'reasonings': reasonings}
        opt_state = self._optimizer.init(params)

        # proto_labels not meaningful for CBC, but needed by base class
        proto_labels = jnp.zeros(self.n_components, dtype=jnp.int32)

        from prosemble.models.prototype_base import SupervisedState
        state = SupervisedState(
            prototypes=components,
            opt_state=opt_state,
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def _compute_loss(self, params, X, y, proto_labels):
        components = params['components']
        reasonings = params['reasonings']

        # Compute similarities (detections)
        diff = X[:, None, :] - components[None, :, :]  # (n, c, d)
        dist_sq = jnp.sum(diff ** 2, axis=2)  # (n, c)
        detections = gaussian_similarity(dist_sq, variance=self.sigma ** 2)

        # Compute class probabilities via CBC reasoning
        class_probs = cbcc(detections, reasonings)  # (n, n_classes)

        # Margin loss
        y_one_hot = jax.nn.one_hot(y, self._n_classes)
        return margin_loss(class_probs, y_one_hot, margin=self.margin)

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        best_params = kwargs.get('best_params')
        if getattr(self, 'restore_best', False) and best_params is not None:
            params = best_params
        self.prototypes_ = params['components']
        self.components_ = params['components']
        self.reasonings_ = params['reasonings']
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
        """Predict class labels."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_cbc_jit(
            X, self.components_, self.reasonings_, self.sigma ** 2
        )

    def predict_proba(self, X):
        """Predict class probabilities via CBC reasoning."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_proba_cbc_jit(
            X, self.components_, self.reasonings_, self.sigma ** 2
        )

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.reasonings_ is not None:
            attrs.append('reasonings_')
        if self.components_ is not None:
            attrs.append('components_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.reasonings_ is not None:
            arrays['reasonings_'] = np.asarray(self.reasonings_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'reasonings_' in arrays:
            self.reasonings_ = jnp.asarray(arrays['reasonings_'])
        if self.prototypes_ is not None:
            self.components_ = self.prototypes_

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['n_components'] = self.n_components
        hp['n_classes'] = self._n_classes
        hp['sigma'] = self.sigma
        return hp
