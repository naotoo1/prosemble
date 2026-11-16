"""
One-class Riemannian soft LVQ (OCRiemannianRSLVQ).

The target class is described by the unnormalised density

.. math::

    p_T(x) = \\sum_{k \\in T} \\exp\\left(b_k - \\frac{d^2(x, w_k)}{2\\sigma^2}\\right)

on a Riemannian manifold, learned by contrast with a reference
distribution on the manifold (see :mod:`prosemble.models.oc_soft_mixin`).

References
----------
.. [1] Gutmann, M., & Hyvarinen, A. (2012). Noise-contrastive estimation
       of unnormalized statistical models, with applications to natural
       image statistics. Journal of Machine Learning Research, 13,
       307-361.
.. [2] Seo, S., & Obermayer, K. (2003). Soft Learning Vector
       Quantization. Neural Computation, 15(7), 1589-1604.
"""

from prosemble.models.oc_soft_mixin import OneClassSoftMixin, RiemannianReferenceMixin
from prosemble.models.riemannian_rslvq import RiemannianRSLVQ


class OCRiemannianRSLVQ(RiemannianReferenceMixin, OneClassSoftMixin, RiemannianRSLVQ):
    """One-class Riemannian robust soft LVQ.

    A density model of one class on a manifold, with the distance of
    :class:`RiemannianRSLVQ` and prototypes that move along geodesics.
    It is trained to tell target samples from samples of a reference
    distribution on the manifold: the uniform distribution on SO(n) and
    the Grassmannian, and a Gaussian in the tangent space at a centre of
    the target samples on SPD(n) and in the Poincare ball. The offsets
    :math:`b_k` play the role of normalising constants, which are not
    available in closed form on these spaces, so the model can be fitted
    to target samples alone. Labelled outliers, when given, get
    prototypes of their own.

    Parameters
    ----------
    manifold : SO, SPD, Grassmannian or HyperbolicPoincare
        Riemannian manifold instance defining the geometry.
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
        Parameters of :class:`RiemannianRSLVQ`.

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
    RiemannianRSLVQ : The multi-class model with the same distance.
    """

    def __init__(self, manifold, n_prototypes=3, n_outlier_prototypes=3,
                 target_label=None, noise_ratio=5, noise_scale=2.5, nu=0.05,
                 outlier_weight='balanced', **kwargs):
        super().__init__(
            n_prototypes=n_prototypes,
            n_outlier_prototypes=n_outlier_prototypes,
            target_label=target_label, noise_ratio=noise_ratio,
            noise_scale=noise_scale, nu=nu, outlier_weight=outlier_weight,
            manifold=manifold, **kwargs)
