"""
One-class differentiable kernel soft LVQ (OCDKRSLVQ).

The target class is described by the unnormalised density

.. math::

    p_T(x) = \\sum_{k \\in T} \\exp\\left(b_k - \\frac{d_\\kappa^2(x, w_k)}{2\\sigma^2}\\right)

on the kernel distance of a Gaussian kernel, and is learned by contrast
with a reference distribution (see :mod:`prosemble.models.oc_soft_mixin`).

References
----------
.. [1] Gutmann, M., & Hyvarinen, A. (2012). Noise-contrastive estimation
       of unnormalized statistical models, with applications to natural
       image statistics. Journal of Machine Learning Research, 13,
       307-361.
.. [2] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
"""

from prosemble.models.dk_rslvq import DKRSLVQ
from prosemble.models.oc_soft_mixin import OneClassSoftMixin


class OCDKRSLVQ(OneClassSoftMixin, DKRSLVQ):
    """One-class differentiable kernel robust soft LVQ.

    A density model of one class on the kernel distance of
    :class:`DKRSLVQ`, with one kernel bandwidth per prototype. It is
    trained to tell target samples from samples of a Gaussian reference,
    which makes the offsets :math:`b_k` play the role of normalising
    constants, so it can be fitted to target samples alone. Labelled
    outliers, when given, get prototypes of their own.

    The kernel distance is bounded, so the density does not vanish far
    from the prototypes but levels off at a floor below the density of
    the training samples.

    Parameters
    ----------
    n_prototypes : int
        Number of prototypes of the target class. Default: 3.
    n_outlier_prototypes : int
        Number of prototypes for labelled outliers, if the training
        labels contain any. Default: 3.
    target_label : int, optional
        Label of the target class. Default: the most frequent label.
    noise_ratio : float
        Reference samples drawn per target sample. Default: 5.
    noise_scale : float
        Width of the reference relative to the spread of the target
        samples. Default: 2.5.
    nu : float
        Fraction of the training targets that ``predict`` rejects.
        Default: 0.05.
    outlier_weight : float or 'balanced'
        Weight of a labelled outlier relative to a target sample.
        Default: ``'balanced'``.
    **kwargs
        Parameters of :class:`DKRSLVQ`.

    Attributes
    ----------
    bias_ : array of shape (n_prototypes_total,)
        Learned offsets :math:`b_k`.
    threshold_ : float
        Log density level of the decision boundary.
    prototype_labels_ : array of shape (n_prototypes_total,)
        0 for a prototype of the target class, 1 for one of the labelled
        outliers.

    See Also
    --------
    OneClassSoftMixin : The training cost and the prediction interface.
    DKRSLVQ : The multi-class model with the same distance.
    """
