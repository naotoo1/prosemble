Time Series
===========

Distances that compare two time series point by point, such as the
Euclidean distance, treat a pattern that occurs a little earlier or
later as a different pattern. Dynamic time warping (DTW) first aligns
the two series, matching each time point to one or more time points of
the other in a monotone way, and then compares them.

SoftDTWRSLVQ
------------

A soft prototype classifier for time series. Its prototypes are time
series themselves, learned by gradient descent through the alignment,
so each class is summarised by a few typical curves that can be plotted
next to the data.

The distance is the **soft-DTW divergence** (Blondel, Mensch and Vert,
2021). Soft-DTW (Cuturi and Blondel, 2017) replaces the minimum over
alignments in DTW by a soft minimum with smoothing :math:`\gamma`, which
makes it differentiable:

.. math::

   \operatorname{sdtw}_\gamma(x, y) = -\gamma \log \sum_{A}
       \exp\left(-\langle A, C(x, y) \rangle / \gamma\right),
   \qquad C_{ij} = \|x_i - y_j\|^2,

where :math:`A` runs over the alignment matrices. Soft-DTW is not zero
for equal series; the divergence

.. math::

   D_\gamma(x, y) = \operatorname{sdtw}_\gamma(x, y)
       - \tfrac12 \operatorname{sdtw}_\gamma(x, x)
       - \tfrac12 \operatorname{sdtw}_\gamma(y, y)

is non-negative and zero only for equal series. Prototype :math:`k` has
the score :math:`f_k(x) = -D_\gamma(x, w_k) / 2\sigma^2`, and training
maximises the likelihood ratio of robust soft LVQ, as for the other soft
models.

.. code-block:: python

   from prosemble.models import SoftDTWRSLVQ

   # X: one series per row, z-normalised; y: class labels
   model = SoftDTWRSLVQ(
       gamma=1.0,                   # smoothing of the soft minimum
       n_prototypes_per_class=1,
       max_iter=150,
       lr=0.05,
   )
   model.fit(X_train, y_train)

   labels = model.predict(X_test)
   proba = model.predict_proba(X_test)
   D = model.divergences(X_test)    # (n_samples, n_prototypes)
   curves = model.prototypes_       # one learned series per row

**Multivariate series.** With ``n_channels=C``, a sample of length
:math:`T` is a row of :math:`T \cdot C` values in time-major order
(all channels of the first time point, then of the second, ...).

**Cost.** One evaluation of soft-DTW costs time proportional to the
product of the two lengths, and training keeps the alignment matrices of
all sample–prototype pairs in memory. For long series or many samples
set ``batch_size``. ``band`` restricts alignments to a Sakoe–Chiba band
(a fraction of the series length) and excludes implausible warps.

Functions
---------

:mod:`prosemble.core.soft_dtw` provides ``soft_dtw``, ``dtw``,
``soft_dtw_divergence`` and their pairwise ``*_matrix`` versions, all
differentiable JAX functions.

Related work
------------

Learning vector quantization under hard DTW was studied by Jain and
Schultz (2018), who update prototypes by asymmetric weighted averaging
of warped series. SoftDTWRSLVQ instead learns its prototypes by gradient
descent on the soft-DTW divergence, with the probabilistic cost of
robust soft LVQ.

References
----------

- Cuturi, M., & Blondel, M. (2017). Soft-DTW: a Differentiable Loss
  Function for Time-Series. *Proceedings of the 34th International
  Conference on Machine Learning*, PMLR 70, 894–903.
- Blondel, M., Mensch, A., & Vert, J.-P. (2021). Differentiable
  Divergences Between Time Series. *Proceedings of the 24th
  International Conference on Artificial Intelligence and Statistics*,
  PMLR 130, 3853–3861.
- Jain, B. J., & Schultz, D. (2018). Asymmetric learning vector
  quantization for efficient nearest neighbor classification in
  dynamic time warping spaces. *Pattern Recognition*, 76, 349–366.
