"""
One-Class Differentiating Kernel GMLVQ (OC-DKGMLVQ).

Combines OC-GLVQ's :math:`\\theta`-based hypothesis testing with the
exponential kernel and adaptive matrix
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
.. [2] Staps et al. (2022). Prototype-based One-Class-Classification
       Learning Using Local Representations. IEEE WSOM+ 2022.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.models.oc_glvq import OCGLVQ
from prosemble.core.activations import sigmoid_beta
from prosemble.core.initializers import identity_omega_init
from prosemble.core.kernel import exponential_kernel_distance_squared


class OCDKGMLVQ(OCGLVQ):
    """One-Class Differentiating Kernel GMLVQ.

    Combines OC-GLVQ with exponential kernel distance and a learnable
    global transformation matrix :math:`\\hat\\Omega` (d x latent_dim).

    .. math::

        d_\\kappa^2(x, w) = \\exp(x^T \\hat\\Lambda x)
                          + \\exp(w^T \\hat\\Lambda w)
                          - 2 \\exp(x^T \\hat\\Lambda w)

    where :math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T`.

    Note: :math:`\\kappa(v, v) \\neq 1` for the exponential kernel, so
    distances are not bounded in :math:`[0, 2]`.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of the transformation. If None, uses input dim.
    omega_hat_scale : float
        Scale factor for omega_hat initialization. Default: 0.1.
        Smaller values prevent exp overflow at initialization.
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
    thetas_ : array of shape (n_prototypes,)
        Learned per-prototype visibility thresholds in kernel distance scale.
    omega_hat_ : array of shape (n_features, latent_dim)
        Learned transformation matrix.

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.
    .. [2] Staps et al. (2022). Prototype-based One-Class-Classification
           Learning Using Local Representations. IEEE WSOM+ 2022.

    See Also
    --------
    OCGLVQ : Base class with Euclidean distance.
    DKGMLVQ : Supervised variant with exponential kernel distance.
    """

    def __init__(self, latent_dim=None, omega_hat_scale=0.1,
                 n_prototypes=3, target_label=None, beta=10.0,
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
        self.latent_dim = latent_dim
        self.omega_hat_scale = omega_hat_scale
        self.omega_hat_ = None

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['omega_hat'] = self.omega_hat_
        return params

    def _init_state(self, X, y, key):
        # Get base OCGLVQ state (prototypes + Euclidean thetas)
        state, params, proto_labels = super()._init_state(X, y, key)

        n_features = X.shape[1]
        latent_dim = self.latent_dim or n_features
        target_mask = (y == self._target_label)
        X_target = X[target_mask]
        prototypes = params['prototypes']

        # Initialize omega_hat (scaled identity)
        omega_hat = self.omega_hat_scale * identity_omega_init(
            n_features, latent_dim
        )

        # Re-initialize thetas using exponential kernel distances
        kernel_dists = exponential_kernel_distance_squared(
            X_target, prototypes, omega_hat
        )
        thetas = jnp.sqrt(jnp.mean(kernel_dists, axis=0) + 1e-10)

        # Update params
        params['omega_hat'] = omega_hat
        params['thetas'] = thetas

        # Re-initialize optimizer
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
        omega_hat = params['omega_hat']

        # Exponential kernel distances: (n, K)
        distances = exponential_kernel_distance_squared(
            X, prototypes, omega_hat
        )

        # OC-GLVQ mu with kernel distance
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
        self.omega_hat_ = params['omega_hat']

    def decision_function(self, X):
        """Compute target-likeness scores using exponential kernel distance.

        Scores near 1.0 indicate target class, near 0.0 indicate outlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : array of shape (n_samples,)
        """
        self._check_fitted()
        X = jnp.asarray(X, dtype=jnp.float32)

        distances = exponential_kernel_distance_squared(
            X, self.prototypes_, self.omega_hat_
        )
        n = X.shape[0]
        nearest_idx = jnp.argmin(distances, axis=1)
        d_nearest = distances[jnp.arange(n), nearest_idx]
        theta_nearest = self.thetas_[nearest_idx]

        mu = (d_nearest - theta_nearest) / (d_nearest + theta_nearest + 1e-10)
        return 1.0 - jax.nn.sigmoid(self.beta * mu)

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
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'omega_hat_' in arrays:
            self.omega_hat_ = jnp.asarray(arrays['omega_hat_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['omega_hat_scale'] = self.omega_hat_scale
        if self.latent_dim is not None:
            hp['latent_dim'] = self.latent_dim
        return hp
