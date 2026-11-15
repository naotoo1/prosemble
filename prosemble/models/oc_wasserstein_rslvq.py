"""
One-class Wasserstein soft LVQ (OCWRSLVQ).

The target class is described by the unnormalised density

.. math::

    p_T(x) = \\sum_{k \\in T} \\exp\\left(b_k - \\frac{W_2^2(x, k)}{2\\sigma^2}\\right)

on the 2-Wasserstein distance to Gaussian prototypes, and is learned by
contrast with a reference distribution (see
:mod:`prosemble.models.oc_soft_mixin`).

References
----------
.. [1] Gutmann, M., & Hyvarinen, A. (2012). Noise-contrastive estimation
       of unnormalized statistical models, with applications to natural
       image statistics. Journal of Machine Learning Research, 13,
       307-361.
.. [2] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
"""

from prosemble.models.oc_soft_mixin import OneClassSoftMixin
from prosemble.models.wasserstein_rslvq import WRSLVQ


class OCWRSLVQ(OneClassSoftMixin, WRSLVQ):
    """One-class Wasserstein robust soft LVQ.

    A density model of one class on the 2-Wasserstein distance of
    :class:`WRSLVQ`, with Gaussian prototypes. It is trained to tell
    target samples from samples of a Gaussian reference, which makes the
    offsets :math:`b_k` play the role of normalising constants, so it
    can be fitted to target samples alone. Labelled outliers, when
    given, get prototypes of their own.

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
        Parameters of :class:`WRSLVQ`.

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
    WRSLVQ : The multi-class model with the same distance.
    """
