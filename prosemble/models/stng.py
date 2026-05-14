"""
Supervised Tangent Neural Gas (STNG).

Combines GTLVQ's per-prototype tangent subspaces with Neural Gas
neighborhood cooperation. Each prototype has an orthonormal basis
defining a tangent subspace; the tangent distance measures distance
in the orthogonal complement. Neighborhood cooperation ensures
robust prototype placement.

Cost function:
    E_STNG = (1/N) sum_mu sum_{r: c(w_r)=c(x_mu)}
        [h(rank_r, gamma) / C(gamma)] * Phi(mu_r)

where:
    d(x, w_r) = ||(I - Omega_r Omega_r^T)(x - w_r)||^2  (tangent distance)
    mu_r = (d_r - d_r^-) / (d_r + d_r^-)
    h(rank, gamma) = exp(-rank / gamma)

References
----------
.. [1] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
.. [2] Saralajew, S., & Villmann, T. (2016). Adaptive tangent
       distances in generalized learning vector quantization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.initializers import random_omega_init
from prosemble.core.utils import orthogonalize


@jit
def _predict_stng_jit(X, prototypes, omegas, proto_labels):
    """JIT-compiled STNG prediction with tangent distance."""
    diff = X[:, None, :] - prototypes[None, :, :]
    proj = jnp.einsum('npd,pds->nps', diff, omegas)
    recon = jnp.einsum('nps,pds->npd', proj, omegas)
    tang_diff = diff - recon
    distances = jnp.sum(tang_diff ** 2, axis=2)
    return wtac(distances, proto_labels)


class STNG(SupervisedPrototypeModel):
    """Supervised Tangent Neural Gas.

    Combines three key ideas:

    - GLVQ loss: (d+ - d-) / (d+ + d-) for margin-based classification
    - Neural Gas cooperation: all same-class prototypes participate in
      the loss, weighted by rank via exp(-rank / gamma)
    - Tangent subspaces: d(x, w_k) = ||(I - Omega_k Omega_k^T)(x - w_k)||^2
      measures distance in the orthogonal complement of each prototype's
      learned invariance subspace

    The neighborhood range gamma decays during training from gamma_init
    to gamma_final. When gamma -> 0, STNG recovers standard GTLVQ.

    Parameters
    ----------
    subspace_dim : int
        Dimension of each prototype's tangent subspace.
    beta : float
        Transfer function steepness parameter for sigmoid shaping.
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

    def __init__(self, subspace_dim=2, beta=10.0, gamma_init=None,
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
        self.subspace_dim = subspace_dim
        self.beta = beta
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.omegas_ = None
        self.gamma_ = None

        # Ensure gamma is frozen from optimizer (not trainable)
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
        key1, key2 = jax.random.split(key)

        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1
        )
        n_protos = prototypes.shape[0]

        # Initialize each Omega as random orthogonal
        keys = jax.random.split(key2, n_protos)
        omegas = jnp.stack([
            random_omega_init(n_features, self.subspace_dim, k) for k in keys
        ])  # (p, d, subspace_dim)

        # Compute gamma_init from prototype count if not set
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)
        gamma_init = self.gamma_init if self.gamma_init is not None else max_per_class / 2.0
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        # Compute decay factor
        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / self.max_iter)

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
        omegas = params['omegas']  # (p, d, s)
        gamma = params['gamma']

        # 1. Tangent distance: d(x, w_k) = ||(I - Omega_k Omega_k^T)(x - w_k)||^2
        diff = X[:, None, :] - prototypes[None, :, :]  # (n, p, d)
        proj_onto_subspace = jnp.einsum('npd,pds->nps', diff, omegas)  # (n, p, s)
        reconstruction = jnp.einsum('nps,pds->npd', proj_onto_subspace, omegas)  # (n, p, d)
        tangent_diff = diff - reconstruction  # (n, p, d)
        distances = jnp.sum(tangent_diff ** 2, axis=2)  # (n, p)

        # 2. Compute ranks within same-class prototypes
        same_class = (y[:, None] == proto_labels[None, :])  # (n, p)
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)  # (n, p)

        # Double argsort for ranks
        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)  # (n, p)

        # 3. Neighborhood function h = exp(-rank / gamma)
        h = jnp.exp(-ranks / (gamma + 1e-10))  # (n, p)
        h = jnp.where(same_class, h, 0.0)  # zero for wrong-class

        # 4. Normalize: C = sum of h over same-class prototypes per sample
        C = jnp.sum(h, axis=1, keepdims=True)  # (n, 1)
        h_normalized = h / (C + 1e-10)  # (n, p)

        # 5. Closest different-class prototype distance
        d_diff = jnp.where(~same_class, distances, INF)
        dm = jnp.min(d_diff, axis=1)  # (n,)

        # 6. GLVQ mu for each (sample, same-class prototype) pair
        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)  # (n, p)

        # 7. Apply transfer function
        from prosemble.core.activations import sigmoid_beta
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)  # (n, p)

        # 8. Rank-weighted sum over same-class prototypes, then mean over samples
        weighted_cost = jnp.sum(h_normalized * cost, axis=1)  # (n,)
        return jnp.mean(weighted_cost)

    def _post_update(self, params):
        # Decay gamma AND re-orthogonalize tangent bases
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        omegas = jax.vmap(orthogonalize)(params['omegas'])
        return {**params, 'gamma': new_gamma, 'omegas': omegas}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omegas_ = params['omegas']
        self.gamma_ = float(params['gamma'])

    def predict(self, X):
        """Predict using tangent distance."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_stng_jit(
            X, self.prototypes_, self.omegas_, self.prototype_labels_
        )

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
        hp['subspace_dim'] = self.subspace_dim
        hp['beta'] = self.beta
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        return hp
