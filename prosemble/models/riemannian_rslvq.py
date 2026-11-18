"""
Riemannian Robust Soft LVQ (RiemannianRSLVQ).

Robust soft LVQ for data on a Riemannian manifold. Prototypes are points
of the manifold, the score of a prototype is built on the squared
distance of the manifold,

.. math::

    f_k(x) = -\\frac{d^2(x, w_k)}{2\\sigma^2},

and prototypes are moved along geodesics.

References
----------
.. [1] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
.. [2] Tang, F., Feng, H., Tino, P., Si, B., & Ji, D. (2021).
       Probabilistic learning vector quantization on manifold of
       symmetric positive definite matrices. Neural Networks, 142,
       105-118.
"""

import jax
import jax.numpy as jnp
import numpy as np

from prosemble.core.protocols import Manifold
from prosemble.models.prototype_base import SupervisedPrototypeModel, SupervisedState
from prosemble.models.riemannian_srng import RiemannianSRNG
from prosemble.models.soft_mixin import SoftAssignmentMixin


class RiemannianRSLVQ(SoftAssignmentMixin, RiemannianSRNG):
    """Riemannian Robust Soft LVQ.

    A labelled mixture on a manifold, with one component per prototype
    and density :math:`p(x | k) \\propto \\exp(-d^2(x, w_k) / 2\\sigma^2)`,
    trained by maximising the likelihood ratio

    .. math::

        \\sum_i \\log \\frac{\\sum_{k: c(k) = y_i} \\exp(-d^2(x_i, w_k) / 2\\sigma^2)}
                           {\\sum_k \\exp(-d^2(x_i, w_k) / 2\\sigma^2)}.

    On SPD matrices with the affine-invariant metric this is the model of
    Tang et al. (2021); here it is available on every supported manifold,
    and the width :math:`\\sigma` can be learned.

    The distance is the one the other supervised Riemannian models train
    with: the affine-invariant, principal-angle and hyperbolic geodesic
    distances on SPD(n), the Grassmannian and the Poincare ball, and the
    chordal distance :math:`\\|R - S\\|_F` on SO(n). With the chordal
    distance each component is :math:`\\propto \\exp(\\operatorname{tr}(w_k^T x) / \\sigma^2)`,
    a matrix Fisher distribution centred at the prototype.

    All four spaces are homogeneous, so the normalising constant of a
    component depends on :math:`\\sigma` but not on the prototype. With
    one shared :math:`\\sigma` it cancels in the likelihood ratio.

    The likelihood ratio is at most 1, and on training data that the
    model separates it keeps growing as the prototypes move away from the
    data along directions that separate the classes: the margin
    :math:`(d^2(x, w_j) - d^2(x, w_k)) / 2\\sigma^2` grows without bound, so
    the cost has no finite minimiser and gradient descent drifts. On small
    training sets the drift acts like a margin-maximising bias and helps
    accuracy. On SPD(n), however, a geodesic distance :math:`t` stretches
    the eigenvalues by :math:`e^{\\pm t}`, and after a long run with a large
    ``lr`` the prototypes can become too ill-conditioned for float32
    arithmetic; training then stops with a warning and keeps the parameters
    with the lowest training loss. The drifted prototypes are also no longer
    typical matrices of their classes.

    With ``prototype_prior=True`` each prototype has a Riemannian Gaussian
    prior centred at the Karcher mean of its class, with the mean squared
    distance of the class to that mean as variance, and the cost is the
    negative log posterior

    .. math::

        -\\frac{1}{n} \\Big( \\sum_i \\log P(y_i \\mid x_i)
        - \\sum_k \\frac{d^2(w_k, m_{c(k)})}{2 s_{c(k)}^2} \\Big),

    whose prior term grows quadratically while the likelihood ratio stays
    bounded, so the cost has a finite minimiser and the prototypes stay near
    their classes, where they can be read as typical class members. On a
    simulated SPD(6) task this was slightly more accurate with 40 training
    matrices per class and less accurate with 5, so the prior is off by
    default.

    Prototypes are updated intrinsically (see :class:`RiemannianSRNG`):
    the optimizer step is taken in the tangent space and followed along
    the geodesic.

    Parameters
    ----------
    manifold : SO, SPD, Grassmannian or HyperbolicPoincare
        Riemannian manifold instance defining the geometry.
    sigma : float or 'auto'
        Initial width :math:`\\sigma` of the mixture components.
        ``'auto'`` (default) sets :math:`\\sigma^2` to the mean squared
        distance from a training sample to its closest prototype of the
        same class at initialisation.
    adapt_sigma : bool
        If True (default), learn :math:`\\sigma` by gradient descent on
        the cost. If False, keep it at its initial value.
    rejection_confidence : float, optional
        Minimum class posterior for ``predict_with_rejection``.
        Default: None (no rejection).
    prototype_prior : bool
        If True, add the Riemannian Gaussian prior that keeps each prototype
        near the Karcher mean of its class (see above). Default: False (the
        plain likelihood-ratio cost).
    tau : float
        Safety factor in (0, 1]. With ``update_rule='riemannian'`` the
        geodesic length of a prototype step is limited to ``tau`` times
        the injectivity radius of the manifold. Default: 0.95.
    update_rule : {'riemannian', 'projected'}
        How prototypes are moved. ``'riemannian'`` (default) takes the
        optimizer step in the tangent space, using the Riemannian
        gradient, and follows the geodesic with the exponential map.
        ``'projected'`` takes a Euclidean step and projects the result
        back onto the manifold.
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
    callbacks : list, optional
        List of Callback objects.
    use_scan : bool
        If True, use jax.lax.scan for training.
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
    gradient_checkpointing : bool, optional
        If True, applies ``jax.checkpoint`` (remat) to the loss function.
        Trades compute for memory by re-computing forward activations during
        the backward pass. Default: False.
    devices : list of jax.Device or None, optional
        Devices for data-parallel training. Data is sharded across devices
        along the batch dimension; params and optimizer state are replicated.
        When using mini-batch training, ``batch_size`` must be divisible by
        the number of devices. Default: None (single-device).

    Attributes
    ----------
    sigma_ : float
        Learned width :math:`\\sigma` of the mixture.

    References
    ----------
    .. [1] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
           Quantization. Neural Computation, 15(7), 1589-1604.
    .. [2] Tang, F., Feng, H., Tino, P., Si, B., & Ji, D. (2021).
           Probabilistic learning vector quantization on manifold of
           symmetric positive definite matrices. Neural Networks, 142,
           105-118.

    See Also
    --------
    RiemannianSRNG : Margin-based counterpart with neural-gas cooperation.
    """

    _hyperparams = ('prototype_prior',)
    _divergence_hint = ('prototype_prior=True keeps the prototypes near '
                        'their classes, which prevents it.')

    def __init__(self, manifold: Manifold, sigma='auto', adapt_sigma=True,
                 rejection_confidence=None, prototype_prior=False, tau=0.95,
                 update_rule='riemannian',
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, optimizer='adam',
                 callbacks=None, use_scan=False, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None,
                 gradient_checkpointing=False, devices=None):
        super().__init__(
            manifold=manifold, tau=tau, update_rule=update_rule,
            n_prototypes_per_class=n_prototypes_per_class,
            max_iter=max_iter, lr=lr, epsilon=epsilon,
            random_seed=random_seed, optimizer=optimizer,
            callbacks=callbacks, use_scan=use_scan, batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            prototypes_initializer=prototypes_initializer,
            patience=patience, restore_best=restore_best,
            class_weight=class_weight,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ema_decay=ema_decay, freeze_params=freeze_params,
            lookahead=lookahead, mixed_precision=mixed_precision,
            gradient_checkpointing=gradient_checkpointing, devices=devices,
        )
        # The parent freezes its neighbourhood range, which this model
        # does not have.
        self.freeze_params = freeze_params
        self._setup_soft(sigma, adapt_sigma, rejection_confidence)
        self.prototype_prior = bool(prototype_prior)
        self._prior = None

    def _squared_distances(self, params, X):
        prototypes = params['prototypes']
        X_m = self._reshape_to_manifold(X, X.shape[0])
        W_m = self._reshape_to_manifold(prototypes, prototypes.shape[0])
        return self._geodesic_distances(X_m, W_m)

    def _project(self, prototypes):
        n_protos = prototypes.shape[0]
        on_manifold = jax.vmap(self.manifold.project)(
            self._reshape_to_manifold(prototypes, n_protos))
        return on_manifold.reshape(n_protos, -1)

    def _init_state(self, X, y, key):
        key1, _ = jax.random.split(key)
        prototypes, proto_labels = self._init_prototypes(
            X, y, self.n_prototypes_per_class, key1)
        prototypes = self._project(prototypes)
        params = self._add_metric_params({'prototypes': prototypes}, X.shape[1])
        params['log_sigma'] = self._initial_log_sigma(
            self._squared_distances(params, X), y, proto_labels)
        self._prior = self._class_prior(X, y, proto_labels) if self.prototype_prior else None
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=self._optimizer.init(params),
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

    def fit(self, X, y, *args, **kwargs):
        """Fit the model (see :meth:`SupervisedPrototypeModel.fit`)."""
        if (kwargs.get('resume') and self.prototype_prior and self._prior is None
                and self.prototypes_ is not None):
            # A loaded model resumes with the prior of the data it is given.
            self._prior = self._class_prior(
                jnp.asarray(X, dtype=jnp.float32), self._encode_y(y), self.prototype_labels_)
        return super().fit(X, y, *args, **kwargs)

    def _karcher_mean(self, points, n_iter=20):
        """Karcher mean of manifold points, by the fixed-point iteration
        :math:`m \\leftarrow \\operatorname{Exp}_m(\\tfrac{1}{n} \\sum_i \\operatorname{Log}_m(x_i))`."""
        m = points[0]
        for _ in range(n_iter):
            step = jnp.mean(jax.vmap(lambda p: self.manifold.log_map(m, p))(points), axis=0)
            m = self.manifold.project(self.manifold.exp_map(m, step))
        return m

    def _class_prior(self, X, y, proto_labels):
        """Centre and variance of the prior of each prototype, from its class."""
        y, labels = np.asarray(y), np.asarray(proto_labels)
        centres, variances = {}, {}
        for c in np.unique(labels):
            members = X[y == c]
            pts = self._reshape_to_manifold(members, members.shape[0])
            m = self._karcher_mean(pts)
            centres[c] = m.reshape(-1)
            d2 = self._geodesic_distances(pts, m[None])[:, 0]
            variances[c] = jnp.maximum(jnp.mean(d2), 1e-6)
        return (jnp.stack([centres[c] for c in labels]),
                jnp.stack([variances[c] for c in labels]),
                float(X.shape[0]))

    def _compute_loss(self, params, X, y, proto_labels):
        loss = super()._compute_loss(params, X, y, proto_labels)
        if self._prior is None:
            return loss
        centres, variances, n_train = self._prior
        n_protos = centres.shape[0]
        d2 = jnp.diagonal(self._geodesic_distances(
            self._reshape_to_manifold(params['prototypes'], n_protos),
            self._reshape_to_manifold(centres, n_protos)))
        return loss + jnp.sum(d2 / (2.0 * variances)) / n_train

    def _logits(self, params, X):
        return self._scale_by_sigma(
            self._squared_distances(params, X), params['log_sigma'])

    def _post_update(self, params):
        return {**params, 'prototypes': self._project(params['prototypes'])}

    def _get_resume_params(self, params):
        return {'prototypes': params['prototypes'],
                'log_sigma': jnp.log(self.sigma_)}

    def _extract_results(self, params, proto_labels, loss_history, n_iter, **kwargs):
        SupervisedPrototypeModel._extract_results(
            self, params, proto_labels, loss_history, n_iter, **kwargs)
        self.sigma_ = jnp.exp(params['log_sigma'])

    def _fitted_params(self):
        return {'prototypes': self.prototypes_, 'log_sigma': jnp.log(self.sigma_)}

    def _get_hyperparams(self):
        hp = super()._get_hyperparams()
        # Neighbourhood parameters of the parent, absent from this model.
        for unused in ('gamma_init', 'gamma_final', 'gamma_decay', 'lr_ratio'):
            hp.pop(unused, None)
        return hp
