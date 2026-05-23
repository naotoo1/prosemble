"""
One-Class Differentiating Kernel GMLVQ with Neural Gas cooperation (OC-DKGMLVQ-NG).

Combines OC-DKGMLVQ's exponential kernel distance and adaptive matrix
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
.. [2] Staps et al. (2022). Prototype-based One-Class-Classification
       Learning Using Local Representations. IEEE WSOM+ 2022.
.. [3] Martinetz, T., Berkovich, S., & Schulten, K. (1993).
       Neural-gas network for the quantization of continuous input
       spaces. IEEE Transactions on Neural Networks.
"""

from prosemble.models.oc_glvq_ng_mixin import NGCooperationMixin
from prosemble.models.oc_dk_matrix_lvq import OCDKGMLVQ
from prosemble.core.kernel import exponential_kernel_distance_squared


class OCDKGMLVQ_NG(NGCooperationMixin, OCDKGMLVQ):
    """One-Class Differentiating Kernel GMLVQ with Neural Gas cooperation.

    Combines OC-DKGMLVQ (exponential kernel distance with learnable
    :math:`\\hat\\Omega` matrix) with NG rank-weighted loss.

    .. math::

        d_\\kappa^2(x, w) = \\exp(x^T \\hat\\Lambda x)
                          + \\exp(w^T \\hat\\Lambda w)
                          - 2 \\exp(x^T \\hat\\Lambda w)

    where :math:`\\hat\\Lambda = \\hat\\Omega \\hat\\Omega^T`.

    Parameters
    ----------
    latent_dim : int, optional
        Dimensionality of the transformation. If None, uses input dim.
    omega_hat_scale : float
        Scale factor for omega_hat initialization. Default: 0.1.
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
    omega_hat_ : array of shape (n_features, latent_dim)
        Learned transformation matrix.
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
    OCDKGMLVQ : Base class without NG cooperation.
    OCGMLVQ_NG : NG variant with Euclidean distance and Omega projection.
    """

    def _compute_distances(self, params, X):
        return exponential_kernel_distance_squared(
            X, params['prototypes'], params['omega_hat']
        )
