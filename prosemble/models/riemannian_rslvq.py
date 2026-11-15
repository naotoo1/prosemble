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

    def __init__(self, manifold: Manifold, sigma='auto', adapt_sigma=True,
                 rejection_confidence=None, tau=0.95,
                 update_rule='riemannian',
                 n_prototypes_per_class=1, max_iter=100, lr=0.01,
                 epsilon=1e-6, random_seed=42, optimizer='adam',
                 callbacks=None, use_scan=False, batch_size=None,
                 lr_scheduler=None, lr_scheduler_kwargs=None,
                 prototypes_initializer=None, patience=None,
                 restore_best=False, class_weight=None,
                 gradient_accumulation_steps=None, ema_decay=None,
                 freeze_params=None, lookahead=None,
                 mixed_precision=None):
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
        )
        # The parent freezes its neighbourhood range, which this model
        # does not have.
        self.freeze_params = freeze_params
        self._setup_soft(sigma, adapt_sigma, rejection_confidence)

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
        state = SupervisedState(
            prototypes=prototypes,
            opt_state=self._optimizer.init(params),
            loss=jnp.array(float('inf')),
            iteration=0,
            converged=False,
        )
        return state, params, proto_labels

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
