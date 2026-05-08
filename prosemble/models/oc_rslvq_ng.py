"""
One-Class RSLVQ with Neural Gas cooperation (OC-RSLVQ-NG).

Combines OC-RSLVQ's probabilistic soft-weighting with Neural Gas
rank-based neighborhood cooperation. Gaussian mixture responsibilities
are modulated by NG neighborhood weights:

    p(k|x) = exp(-d_k / 2σ²) / Σ_j exp(-d_j / 2σ²)
    h_k = exp(-rank_k / γ)
    w_k = p(k|x) · h_k / Σ_j p(j|x) · h_j
    loss = mean(Σ_k w_k · sigmoid(μ_k + margin, β))

Uses standard Euclidean distance (no metric adaptation).

References
----------
.. [1] Seo, S., & Obermayer, K. (2003). Soft Nearest Prototype
       Classification. IEEE Trans. Neural Networks, 15(7):1589-1604.
.. [2] Hammer, B., Strickert, M., & Villmann, T. (2003). Supervised
       Neural Gas with General Similarity Measure. Neural Processing
       Letters.
.. [3] Staps et al. (2022). Prototype-based One-Class-Classification
       Learning Using Local Representations. IJCNN 2022.
"""

from prosemble.models.oc_rslvq_ng_mixin import OCRSLVQNGMixin
from prosemble.models.oc_rslvq import OCRSLVQ


class OCRSLVQ_NG(OCRSLVQNGMixin, OCRSLVQ):
    """One-Class RSLVQ with Neural Gas neighborhood cooperation.

    Combines soft Gaussian mixture responsibilities with NG rank-based
    cooperation. Uses standard Euclidean distances.

    Parameters
    ----------
    sigma : float
        Bandwidth of Gaussian mixture for prototype weighting.
    n_prototypes : int
        Number of prototypes for the target class.
    target_label : int, optional
        Target (normal) class label. Default: auto-detect.
    beta : float
        Sigmoid steepness. Default: 10.0.
    gamma_init : float, optional
        Initial neighborhood range. Default: n_prototypes / 2.
    gamma_final : float
        Final neighborhood range. Default: 0.01.
    gamma_decay : float, optional
        Per-step multiplicative decay for gamma.

    Attributes
    ----------
    thetas_ : array of shape (n_prototypes,)
        Learned per-prototype acceptance thresholds.
    gamma_ : float
        Final gamma value after training.
    """

    def _compute_distances(self, params, X):
        return self.distance_fn(X, params['prototypes'])
