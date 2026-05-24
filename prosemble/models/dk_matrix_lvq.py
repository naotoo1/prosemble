"""
Differentiating Kernel GMLVQ (DKGMLVQ).

GMLVQ with the exponential kernel and adaptive matrix
:math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T`:

.. math::

    \\kappa_{\\exp}(x, w, \\hat\\Lambda) = \\exp(x^T \\hat\\Lambda w)

.. math::

    d_\\kappa^2(x, w) = \\exp(x^T \\hat\\Lambda x)
                      + \\exp(w^T \\hat\\Lambda w)
                      - 2 \\exp(x^T \\hat\\Lambda w)

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
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
def _predict_dkgmlvq_jit(X, prototypes, omega_hat, proto_labels):
    """JIT-compiled DKGMLVQ prediction with exponential kernel distance."""
    distances = exponential_kernel_distance_squared(X, prototypes, omega_hat)
    return wtac(distances, proto_labels)


class DKGMLVQ(SupervisedPrototypeModel):
    """Differentiating Kernel GMLVQ with Exponential Kernel.

    Learns a global transformation matrix :math:`\\hat\\Omega` (d x latent_dim)
    such that distances are computed via the exponential kernel:

    .. math::

        \\kappa_{\\exp}(x, w) = \\exp(x^T \\hat\\Lambda w),
        \\quad \\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T

    .. math::

        d_\\kappa^2(x, w) = \\exp(x^T \\hat\\Lambda x)
                          + \\exp(w^T \\hat\\Lambda w)
                          - 2 \\exp(x^T \\hat\\Lambda w)

    Note: :math:`\\kappa(v, v) \\neq 1` for the exponential kernel, so the
    full three-term distance formula is used.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of the transformation. If None, uses input dim.
    omega_hat_scale : float
        Scale factor for omega_hat initialization. Default: 0.1.
        Smaller values prevent exp overflow at initialization.
    beta : float
        Transfer function steepness.
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

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.

    See Also
    --------
    SupervisedPrototypeModel : Full list of base parameters.
    """

    def __init__(self, latent_dim=None, omega_hat_scale=0.1, beta=10.0,
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, distance_fn=None,
                 optimizer='adam', transfer_fn=None, margin=0.0,
                 callbacks=None, use_scan=True, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None, mixed_precision=None,
                 **kwargs):
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
            **kwargs,
        )
        self.latent_dim = latent_dim
        self.omega_hat_scale = omega_hat_scale
        self.beta = beta
        self.omega_hat_ = None

    def _get_resume_params(self, params):
        params['omega_hat'] = self.omega_hat_
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

        params = {'prototypes': prototypes, 'omega_hat': omega_hat}
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
        distances = exponential_kernel_distance_squared(
            X, prototypes, omega_hat
        )
        from prosemble.core.losses import glvq_loss_with_transfer
        return glvq_loss_with_transfer(
            distances, y, proto_labels,
            transfer_fn=self.transfer_fn,
            margin=self.margin,
            beta=self.beta,
        )

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        super()._extract_results(params, proto_labels, loss_history, n_iter, **kwargs)
        self.omega_hat_ = params['omega_hat']

    @property
    def omega_hat_matrix(self):
        """Return the learned :math:`\\hat\\Omega` matrix."""
        if self.omega_hat_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_hat_

    @property
    def lambda_hat_matrix(self):
        """Return :math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T`."""
        if self.omega_hat_ is None:
            raise ValueError("Model not fitted.")
        return self.omega_hat_ @ self.omega_hat_.T

    def _compute_distances_for_rejection(self, X):
        """Exponential kernel distances for reject option."""
        return exponential_kernel_distance_squared(X, self.prototypes_, self.omega_hat_)

    def predict(self, X):
        """Predict using learned exponential kernel distance."""
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)
        return _predict_dkgmlvq_jit(
            X, self.prototypes_, self.omega_hat_, self.prototype_labels_
        )

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.omega_hat_ is not None:
            attrs.append('omega_hat_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.omega_hat_ is not None:
            arrays['omega_hat_'] = np.asarray(self.omega_hat_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omega_hat_' in arrays:
            self.omega_hat_ = jnp.asarray(arrays['omega_hat_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['beta'] = self.beta
        hp['omega_hat_scale'] = self.omega_hat_scale
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
