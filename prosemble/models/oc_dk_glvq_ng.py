"""
One-Class Differentiating Kernel GLVQ with Neural Gas cooperation (OC-DKGLVQ-NG).

Combines OC-DKGLVQ's Gaussian kernel distance and learnable per-prototype
bandwidths with Neural Gas neighborhood cooperation. All prototypes
participate in the loss, weighted by their distance rank.

.. math::

    d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
        -\\frac{\\|x - w_k\\|^2}{2\\sigma_k^2}
    \\right)\\right)

References
----------
.. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
       quantization in gradient-descent learning. Neurocomputing.
.. [2] Staps et al. (2022). Prototype-based One-Class-Classification
       Learning Using Local Representations. IEEE WSOM+ 2022.
.. [3] Martinetz, T., Berkovich, S., & Schulten, K. (1993).
       Neural-gas network for the quantization of continuous input
       spaces. IEEE Transactions on Neural Networks.
"""

import jax.numpy as jnp

from prosemble.models.oc_glvq_ng_mixin import NGCooperationMixin
from prosemble.models.oc_dk_glvq import OCDKGLVQ
from prosemble.core.kernel import kernel_distance_squared_per_proto


class OCDKGLVQ_NG(NGCooperationMixin, OCDKGLVQ):
    """One-Class Differentiating Kernel GLVQ with Neural Gas cooperation.

    Combines OC-DKGLVQ (Gaussian kernel distance with learnable bandwidths)
    with NG rank-weighted loss where all prototypes participate.

    .. math::

        d_\\kappa^2(x, w_k) = 2\\left(1 - \\exp\\left(
            -\\frac{\\|x - w_k\\|^2}{2\\sigma_k^2}
        \\right)\\right)

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
    gamma_init : float, optional
        Initial neighborhood range. Default: n_prototypes / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.
        Default: computed from max_iter so gamma reaches gamma_final.
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
    gamma_ : float
        Final gamma value after training.

    References
    ----------
    .. [1] Villmann, T., Haase, S., & Kaden, M. (2015). Kernelized vector
           quantization in gradient-descent learning. Neurocomputing.
    .. [2] Staps et al. (2022). Prototype-based One-Class-Classification
           Learning Using Local Representations. IEEE WSOM+ 2022.
    .. [3] Martinetz, T., Berkovich, S., & Schulten, K. (1993).
           Neural-gas network for the quantization of continuous input
           spaces. IEEE Transactions on Neural Networks.

    See Also
    --------
    OCDKGLVQ : Base class without NG cooperation.
    OCGLVQ_NG : NG variant with Euclidean distance.
    """

    def _compute_distances(self, params, X):
        sigmas = jnp.maximum(params['sigmas'], self.sigma_min)
        return kernel_distance_squared_per_proto(X, params['prototypes'], sigmas)
