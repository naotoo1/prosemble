"""
One-Class GLVQ (OC-GLVQ).

Adapts GLVQ's hypothesis-testing principle for one-class classification.
In standard GLVQ, the classifier function is:

.. math::

    \\mu = \\frac{d^+ - d^-}{d^+ + d^-}

where :math:`d^+` is distance to nearest same-class prototype and :math:`d^-` is distance
to nearest different-class prototype. For one-class classification there
is no competing class, so we replace :math:`d^-` with a learned per-prototype
visibility threshold :math:`\\theta_k`:

.. math::

    \\mu_{k^*}(x_i) = s_i \\cdot \\frac{d(x_i, w_{k^*}) - \\theta_{k^*}}{d(x_i, w_{k^*}) + \\theta_{k^*}}

where :math:`k^*` is the nearest prototype and :math:`s_i = +1` for target, :math:`-1` for outlier.

- Target with :math:`d < \\theta`: :math:`\\mu < 0 \\to f(\\mu) \\approx 0` -- low cost (correct)
- Target with :math:`d > \\theta`: :math:`\\mu > 0 \\to f(\\mu) \\approx 1` -- high cost (misclassified)
- Outlier with :math:`d > \\theta`: :math:`\\mu < 0 \\to f(\\mu) \\approx 0` -- low cost (correct)
- Outlier with :math:`d < \\theta`: :math:`\\mu > 0 \\to f(\\mu) \\approx 1` -- high cost (misclassified)

The loss is :math:`E = \\text{mean}(f(\\mu + \\text{margin}))` where :math:`f` is a sigmoid transfer.

References
----------
.. [1] Sato, A., & Yamada, K. (1995). Generalized Learning Vector
       Quantization. NIPS.
.. [2] Staps et al. (2022). Prototype-based One-Class-Classification
       Learning Using Local Representations. IEEE WSOM+ 2022.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.initializers import stratified_selection_init
from prosemble.core.activations import sigmoid_beta


class OCGLVQ(SupervisedPrototypeModel):
    """One-Class Generalized Learning Vector Quantization.

    Combines GLVQ's :math:`\\mu`-based hypothesis testing with per-prototype
    visibility thresholds :math:`\\theta_k` for one-class classification.

    Parameters
    ----------
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Which label is the target (normal) class. Default: auto-detect
        as the most frequent class.
    beta : float
        Sigmoid steepness for the transfer function. Default: 10.0.
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
    thetas_ : array of shape (n_prototypes,)
        Learned per-prototype visibility thresholds.
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
            n_prototypes_per_class=n_prototypes,
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
        self.n_prototypes = n_prototypes
        self.target_label = target_label
        self.beta = beta

        # Fitted attributes
        self.thetas_ = None
        self._target_label = None
        self._non_target_label = None

    def _get_resume_params(self, params):
        params['thetas'] = self.thetas_
        return params

    def _init_state(self, X, y, key):
        # Determine target and non-target labels
        classes = jnp.unique(y)
        if self.target_label is not None:
            self._target_label = int(self.target_label)
        else:
            counts = jnp.array([jnp.sum(y == c) for c in classes])
            self._target_label = int(classes[jnp.argmax(counts)])

        non_target = classes[classes != self._target_label]
        self._non_target_label = int(non_target[0]) if len(non_target) > 0 else (
            1 - self._target_label
        )

        # Filter target class data for prototype initialization
        target_mask = (y == self._target_label)
        X_target = X[target_mask]
        y_target = jnp.full(X_target.shape[0], self._target_label, dtype=jnp.int32)

        key1, key2 = jax.random.split(key)

        # Initialize prototypes from target class data
        if self.prototypes_initializer is not None:
            prototypes, _ = self._init_prototypes(
                X_target, y_target, self.n_prototypes, key1
            )
        else:
            prototypes, _ = stratified_selection_init(
                X_target, y_target, self.n_prototypes, key1
            )
        proto_labels = jnp.full(
            self.n_prototypes, self._target_label, dtype=jnp.int32
        )

        # Initialize thetas: sqrt of mean squared distance per prototype
        from prosemble.core.distance import squared_euclidean_distance_matrix
        dists = squared_euclidean_distance_matrix(X_target, prototypes)
        thetas = jnp.sqrt(jnp.mean(dists, axis=0) + 1e-10)

        params = {
            'prototypes': prototypes,
            'thetas': thetas,
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
        thetas = params['thetas']

        # Squared Euclidean distances: (n, K)
        distances = self.distance_fn(X, prototypes)

        # Nearest prototype for each sample
        n = X.shape[0]
        nearest_idx = jnp.argmin(distances, axis=1)
        d_nearest = distances[jnp.arange(n), nearest_idx]
        theta_nearest = thetas[nearest_idx]

        # Signed label: +1 for target, -1 for outlier
        s = jnp.where(y == self._target_label, 1.0, -1.0)

        # OC-GLVQ mu: s * (d - theta) / (d + theta)
        mu = s * (d_nearest - theta_nearest) / (d_nearest + theta_nearest + 1e-10)

        # Apply transfer function
        transfer = self.transfer_fn or sigmoid_beta
        return jnp.mean(transfer(mu + self.margin, self.beta))

    def _post_update(self, params):
        thetas = jnp.maximum(params['thetas'], 1e-6)
        return {**params, 'thetas': thetas}

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.thetas_ = jnp.maximum(params['thetas'], 1e-6)

    def decision_function(self, X):
        """Compute target-likeness scores.

        Scores near 1.0 indicate target class, near 0.0 indicate outlier.
        The decision boundary is at score = 0.5 (where :math:`d = \\theta`).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        distances = self.distance_fn(X, self.prototypes_)
        n = X.shape[0]
        nearest_idx = jnp.argmin(distances, axis=1)
        d_nearest = distances[jnp.arange(n), nearest_idx]
        theta_nearest = self.thetas_[nearest_idx]

        # mu from target perspective (no sign flip)
        mu = (d_nearest - theta_nearest) / (d_nearest + theta_nearest + 1e-10)
        # 1 - sigmoid: d < theta → score > 0.5, d > theta → score < 0.5
        return 1.0 - jax.nn.sigmoid(self.beta * mu)

    def predict(self, X):
        """Predict target or non-target labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        scores = self.decision_function(X)
        return jnp.where(
            scores >= 0.5, self._target_label, self._non_target_label
        ).astype(jnp.int32)

    def predict_with_reject(self, X, upper=0.5, lower=None, reject_label=-1):
        """Predict with a reject option for uncertain samples.

        Samples with scores in [lower, upper) are rejected.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        upper : float
            Scores >= upper are classified as target. Default: 0.5.
        lower : float, optional
            Scores < lower are classified as non-target. Default: same
            as upper (no rejection zone).
        reject_label : int
            Label for rejected samples. Default: -1.

        Returns
        -------
        labels : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        if lower is None:
            lower = upper
        scores = self.decision_function(X)
        labels = jnp.full(scores.shape, reject_label, dtype=jnp.int32)
        labels = jnp.where(scores >= upper, self._target_label, labels)
        labels = jnp.where(scores < lower, self._non_target_label, labels)
        return labels

    def predict_proba(self, X):
        """Predict probability of being target class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return self.decision_function(X)

    @property
    def visibility_radii(self):
        """Return the learned visibility radii :math:`\\theta_k` for each prototype."""
        if self.thetas_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.thetas_

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.thetas_ is not None:
            attrs.append('thetas_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.thetas_ is not None:
            arrays['thetas_'] = np.asarray(self.thetas_)
        if self._target_label is not None:
            arrays['_target_label'] = np.asarray(self._target_label)
        if self._non_target_label is not None:
            arrays['_non_target_label'] = np.asarray(self._non_target_label)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'thetas_' in arrays:
            self.thetas_ = jnp.asarray(arrays['thetas_'])
        if '_target_label' in arrays:
            self._target_label = int(arrays['_target_label'])
        if '_non_target_label' in arrays:
            self._non_target_label = int(arrays['_non_target_label'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp.pop('n_prototypes_per_class', None)
        hp['n_prototypes'] = self.n_prototypes
        hp['target_label'] = self.target_label
        hp['beta'] = self.beta
        return hp
