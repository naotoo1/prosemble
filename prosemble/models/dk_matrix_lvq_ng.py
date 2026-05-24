"""
Differentiating Kernel GMLVQ with Neural Gas cooperation (DKGMLVQ-NG).

Combines DKGMLVQ's exponential kernel distance and adaptive matrix
:math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T` with Neural Gas
neighborhood cooperation.

.. math::

    d_\\kappa^2(x, w) = \\exp(x^T \\hat\\Lambda x)
                      + \\exp(w^T \\hat\\Lambda w)
                      - 2 \\exp(x^T \\hat\\Lambda w)

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np

from prosemble.models.prototype_base import SupervisedPrototypeModel
from prosemble.core.competitions import wtac
from prosemble.core.initializers import identity_omega_init
from prosemble.core.kernel import exponential_kernel_distance_squared


@jit
def _predict_dkgmlvq_ng_jit(X, prototypes, omega_hat, proto_labels):
    """JIT-compiled prediction with exponential kernel distance."""
    distances = exponential_kernel_distance_squared(X, prototypes, omega_hat)
    return wtac(distances, proto_labels)


class DKGMLVQ_NG(SupervisedPrototypeModel):
    """Differentiating Kernel GMLVQ with Neural Gas cooperation.

    Combines exponential kernel distance with a learnable transformation
    matrix :math:`\\hat\\Omega` and Neural Gas rank-weighted loss.

    .. math::

        d_\\kappa^2(x, w) = \\exp(x^T \\hat\\Lambda x)
                          + \\exp(w^T \\hat\\Lambda w)
                          - 2 \\exp(x^T \\hat\\Lambda w)

    where :math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T`.

    When :math:`\\gamma \\to 0`, DKGMLVQ-NG recovers standard DKGMLVQ.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of the transformation. If None, uses input dim.
    omega_hat_scale : float
        Scale factor for omega_hat initialization. Default: 0.1.
    beta : float
        Transfer function steepness. Default: 10.0.
    gamma_init : float, optional
        Initial neighborhood range. Default: max prototypes per class / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.
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
    optimizer : str or optax optimizer, optional
        Optimizer name ('adam', 'sgd') or optax GradientTransformation.
    transfer_fn : callable, optional
        Transfer function for loss shaping (default: identity).
    margin : float
        Margin for loss computation.
    callbacks : list, optional
        List of Callback objects.
    use_scan : bool
        If True (default), use jax.lax.scan for training.
    batch_size : int, optional
        Mini-batch size. If None, use full-batch training.
    lr_scheduler : str or optax.Schedule, optional
        Learning rate schedule.
    lr_scheduler_kwargs : dict, optional
        Keyword arguments for the learning rate scheduler.
    prototypes_initializer : str or callable, optional
        How to initialize prototypes.
    patience : int, optional
        Epochs with no improvement before stopping.
    restore_best : bool
        If True, restore best parameters after training.
    class_weight : dict or 'balanced', optional
        Weights for each class.
    gradient_accumulation_steps : int, optional
        Accumulate gradients over this many steps.
    ema_decay : float, optional
        Exponential moving average decay for parameters.
    freeze_params : list of str, optional
        Parameter group names to freeze.
    lookahead : dict, optional
        Lookahead optimizer wrapper configuration.
    mixed_precision : str or None, optional
        Compute dtype for mixed precision training.

    Attributes
    ----------
    omega_hat_ : array of shape (n_features, latent_dim)
        Learned transformation matrix.
    gamma_ : float
        Final gamma value after training.

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.
    .. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
           Neural Gas with General Similarity Measure. Neural Processing
           Letters.

    See Also
    --------
    DKGMLVQ : Base variant without NG cooperation.
    SMNG : NG variant with Euclidean Omega-projected distance.
    """

    def __init__(self, latent_dim=None, omega_hat_scale=0.1, beta=10.0,
                 gamma_init=None, gamma_final=0.01, gamma_decay=None,
                 n_prototypes_per_class=1, max_iter=100,
                 lr=0.01, epsilon=1e-6, random_seed=42, distance_fn=None,
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
        self.latent_dim = latent_dim
        self.omega_hat_scale = omega_hat_scale
        self.beta = beta
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.gamma_decay = gamma_decay
        self.omega_hat_ = None
        self.gamma_ = None

        # Ensure gamma is frozen from optimizer
        if self.freeze_params is None:
            self.freeze_params = ['gamma']
        elif 'gamma' not in self.freeze_params:
            self.freeze_params = list(self.freeze_params) + ['gamma']

    def _get_resume_params(self, params):
        params['omega_hat'] = self.omega_hat_
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
        omega_hat = self.omega_hat_scale * identity_omega_init(
            n_features, latent_dim
        )

        # Gamma initialization
        if isinstance(self.n_prototypes_per_class, int):
            max_per_class = self.n_prototypes_per_class
        elif isinstance(self.n_prototypes_per_class, dict):
            max_per_class = max(self.n_prototypes_per_class.values())
        else:
            max_per_class = max(self.n_prototypes_per_class)
        gamma_init = self.gamma_init if self.gamma_init is not None else max_per_class / 2.0
        gamma_init = max(gamma_init, self.gamma_final + 1e-6)
        self._gamma_init_actual = gamma_init

        if self.gamma_decay is not None:
            self._gamma_decay = self.gamma_decay
        else:
            self._gamma_decay = (self.gamma_final / gamma_init) ** (1.0 / self.max_iter)

        params = {
            'prototypes': prototypes,
            'omega_hat': omega_hat,
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
        omega_hat = params['omega_hat']
        gamma = params['gamma']

        # Exponential kernel distances: (n, p)
        distances = exponential_kernel_distance_squared(
            X, prototypes, omega_hat
        )

        # Class-aware ranking
        same_class = (y[:, None] == proto_labels[None, :])
        INF = jnp.finfo(distances.dtype).max
        d_same = jnp.where(same_class, distances, INF)

        order = jnp.argsort(d_same, axis=1)
        ranks = jnp.argsort(order, axis=1).astype(jnp.float32)

        h = jnp.exp(-ranks / (gamma + 1e-10))
        h = jnp.where(same_class, h, 0.0)

        C = jnp.sum(h, axis=1, keepdims=True)
        h_normalized = h / (C + 1e-10)

        d_diff = jnp.where(~same_class, distances, INF)
        dm = jnp.min(d_diff, axis=1)

        mu = (distances - dm[:, None]) / (distances + dm[:, None] + 1e-10)

        from prosemble.core.activations import sigmoid_beta
        transfer = self.transfer_fn or sigmoid_beta
        cost = transfer(mu + self.margin, self.beta)

        weighted_cost = jnp.sum(h_normalized * cost, axis=1)
        return jnp.mean(weighted_cost)

    def _post_update(self, params):
        new_gamma = params['gamma'] * self._gamma_decay
        new_gamma = jnp.maximum(new_gamma, self.gamma_final)
        return {**params, 'gamma': new_gamma}

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.omega_hat_ = params['omega_hat']
        self.gamma_ = float(params['gamma'])

    def _compute_distances_for_rejection(self, X):
        """Exponential kernel distances for reject option."""
        return exponential_kernel_distance_squared(X, self.prototypes_, self.omega_hat_)

    def predict(self, X):
        """Predict using learned exponential kernel distance."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_dkgmlvq_ng_jit(
            X, self.prototypes_, self.omega_hat_, self.prototype_labels_
        )

    @property
    def omega_hat_matrix(self):
        """Return the learned :math:`\\hat\\Omega` matrix."""
        if self.omega_hat_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.omega_hat_

    @property
    def lambda_hat_matrix(self):
        """Return :math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T`."""
        if self.omega_hat_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.omega_hat_ @ self.omega_hat_.T

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omega_hat_ is not None:
            attrs.append('omega_hat_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omega_hat_ is not None:
            arrays['omega_hat_'] = np.asarray(self.omega_hat_)
        if self.gamma_ is not None:
            arrays['gamma_'] = np.asarray(self.gamma_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omega_hat_' in arrays:
            self.omega_hat_ = jnp.asarray(arrays['omega_hat_'])
        if 'gamma_' in arrays:
            self.gamma_ = float(arrays['gamma_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        hp['omega_hat_scale'] = self.omega_hat_scale
        hp['gamma_init'] = self.gamma_init
        hp['gamma_final'] = self.gamma_final
        hp['gamma_decay'] = self.gamma_decay
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
