"""
One-Class Differentiating Kernel GRLVQ (OC-DKGRLVQ).

Combines OC-GLVQ's :math:`\\theta`-based hypothesis testing with
per-feature relevance weighting and Gaussian kernel distance:

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{\\sum_j \\lambda_j (x_j - w_{kj})^2}{2\\sigma_k^2}
    \\right)\\right)

where :math:`\\lambda = \\text{softmax}(\\text{relevances})`.

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
from prosemble.core.kernel import kernel_distance_squared_relevance


class OCDKGRLVQ(OCGLVQ):
    """One-Class Differentiating Kernel GRLVQ.

    Combines OC-GLVQ with per-feature relevance weighting and Gaussian
    kernel distance with per-prototype bandwidth adaptation.

    .. math::

        d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
            -\\frac{\\sum_j \\lambda_j (x_j - w_{kj})^2}{2\\sigma_k^2}
        \\right)\\right)

    where :math:`\\lambda = \\text{softmax}(\\text{relevances})` are
    learned per-feature weights.

    Parameters
    ----------
    sigma_init : str or float
        Initialization strategy for per-prototype bandwidths.
        'median' (default): per-prototype median distance from prototype
        to target class members.
        'mean': per-prototype mean distance.
        float: fixed value for all prototypes.
    sigma_min : float
        Lower bound for sigma to prevent bandwidth collapse. Default: 1e-3.
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
    sigmas_ : array of shape (n_prototypes,)
        Learned per-prototype kernel bandwidths.
    relevances_ : array of shape (n_features,)
        Learned per-feature relevance weights (raw logits).

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.
    .. [2] Staps et al. (2022). Prototype-based One-Class-Classification
           Learning Using Local Representations. IEEE WSOM+ 2022.

    See Also
    --------
    OCGLVQ : Base class with Euclidean distance.
    DKGRLVQ : Supervised variant with kernel distance and relevances.
    """

    def __init__(self, sigma_init='median', sigma_min=1e-3,
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
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.sigmas_ = None
        self.relevances_ = None

    def _estimate_sigmas(self, X_target, prototypes):
        """Estimate per-prototype bandwidths from target data."""
        if isinstance(self.sigma_init, (int, float)):
            return jnp.full(prototypes.shape[0], float(self.sigma_init))

        sigmas = []
        for k in range(prototypes.shape[0]):
            dists = jnp.sqrt(jnp.sum((X_target - prototypes[k]) ** 2, axis=1))
            if self.sigma_init == 'median':
                sigma_k = jnp.median(dists)
            else:  # 'mean'
                sigma_k = jnp.mean(dists)
            sigmas.append(jnp.maximum(sigma_k, self.sigma_min))
        return jnp.array(sigmas)

    def _get_resume_params(self, params):
        params = super()._get_resume_params(params)
        params['sigmas'] = self.sigmas_
        params['relevances'] = self.relevances_
        return params

    def _init_state(self, X, y, key):
        # Get base OCGLVQ state (prototypes + Euclidean thetas)
        state, params, proto_labels = super()._init_state(X, y, key)

        n_features = X.shape[1]
        target_mask = (y == self._target_label)
        X_target = X[target_mask]
        prototypes = params['prototypes']

        # Estimate sigmas from target data
        sigmas = self._estimate_sigmas(X_target, prototypes)

        # Initialize relevances (uniform in logit space)
        relevances = jnp.ones(n_features) / n_features

        # Re-initialize thetas using relevance-weighted kernel distances
        lam = jax.nn.softmax(relevances)
        kernel_dists = kernel_distance_squared_relevance(
            X_target, prototypes, sigmas, lam
        )
        thetas = jnp.sqrt(jnp.mean(kernel_dists, axis=0) + 1e-10)

        # Update params
        params['sigmas'] = sigmas
        params['relevances'] = relevances
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
        sigmas = jnp.maximum(params['sigmas'], self.sigma_min)
        lam = jax.nn.softmax(params['relevances'])

        # Relevance-weighted kernel distances: (n, K), bounded in [0, 2]
        distances = kernel_distance_squared_relevance(
            X, prototypes, sigmas, lam
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

    def _post_update(self, params):
        params = super()._post_update(params)  # thetas >= 1e-6
        params['sigmas'] = jnp.maximum(params['sigmas'], self.sigma_min)
        return params

    def _extract_results(self, params, proto_labels, loss_history, n_iter,
                         **kwargs):
        super()._extract_results(
            params, proto_labels, loss_history, n_iter, **kwargs
        )
        self.sigmas_ = params['sigmas']
        self.relevances_ = params['relevances']  # raw logits

    def decision_function(self, X):
        """Compute target-likeness scores using relevance-weighted kernel distance.

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

        sigmas = jnp.maximum(self.sigmas_, self.sigma_min)
        lam = jax.nn.softmax(self.relevances_)
        distances = kernel_distance_squared_relevance(
            X, self.prototypes_, sigmas, lam
        )
        n = X.shape[0]
        nearest_idx = jnp.argmin(distances, axis=1)
        d_nearest = distances[jnp.arange(n), nearest_idx]
        theta_nearest = self.thetas_[nearest_idx]

        mu = (d_nearest - theta_nearest) / (d_nearest + theta_nearest + 1e-10)
        return 1.0 - jax.nn.sigmoid(self.beta * mu)

    @property
    def kernel_bandwidths(self):
        """Return the learned per-prototype bandwidths."""
        if self.sigmas_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.sigmas_

    @property
    def relevance_profile(self):
        """Return the learned relevance weights (normalized via softmax)."""
        if self.relevances_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return jax.nn.softmax(self.relevances_)

    def _get_quantizable_attrs(self):
        attrs = super()._get_quantizable_attrs()
        if self.sigmas_ is not None:
            attrs.append('sigmas_')
        if self.relevances_ is not None:
            attrs.append('relevances_')
        return attrs

    def _get_fitted_arrays(self):
        arrays = super()._get_fitted_arrays()
        if self.sigmas_ is not None:
            arrays['sigmas_'] = np.asarray(self.sigmas_)
        if self.relevances_ is not None:
            arrays['relevances_'] = np.asarray(self.relevances_)
        return arrays

    def _set_fitted_arrays(self, arrays):
        super()._set_fitted_arrays(arrays)
        if 'sigmas_' in arrays:
            self.sigmas_ = jnp.asarray(arrays['sigmas_'])
        if 'relevances_' in arrays:
            self.relevances_ = jnp.asarray(arrays['relevances_'])

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        hp['sigma_init'] = self.sigma_init
        hp['sigma_min'] = self.sigma_min
        return hp
