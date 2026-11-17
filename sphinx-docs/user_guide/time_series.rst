Time series
===========

.. currentmodule:: prosemble.models

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

Multivariate series with a learned metric
-----------------------------------------

With several channels, :class:`SoftDTWRSLVQ` weighs every channel alike:
two time points are compared by the squared Euclidean distance of their
channel values. :class:`SoftDTWMRSLVQ` learns a metric over the channels
together with the prototypes. Inside the alignment, time points are
compared by

.. math::

   c(x_t, w_s) = (x_t - w_s)^\top \Lambda\, (x_t - w_s),
   \qquad \Lambda = \Omega \Omega^\top,

which is the squared Euclidean distance between the mapped points
:math:`\Omega^\top x_t` and :math:`\Omega^\top w_s`. The diagonal of
:math:`\Lambda` is the relevance of each channel, and its off-diagonal
entries show channels that act together, as in :class:`GMLVQ`.
:class:`SoftDTWLMRSLVQ` learns one :math:`\Lambda_k` per prototype.

.. code-block:: python

   from prosemble.models import SoftDTWMRSLVQ

   # X: one series per row, T time points x C channels, time-major
   model = SoftDTWMRSLVQ(
       n_channels=C,
       gamma=float(C),                # one unit of smoothing per channel
       n_prototypes_per_class=1,
       max_iter=150,
       lr=0.05,
   )
   model.fit(X_train, y_train)

   model.channel_relevances          # (C,), non-negative, sums to 1
   model.lambda_matrix               # (C, C)

The scale of :math:`\Omega` would also scale the ground cost, and with it
the meaning of ``gamma``. The map is therefore normalised to
:math:`\operatorname{tr}\Lambda = C`: training starts from
:math:`\Lambda = I`, which is :class:`SoftDTWRSLVQ`, and only redistributes
weight between the channels. As in the other matrix models, the matrix
learns more slowly than the prototypes (``matrix_lr_ratio=0.3``), and
``latent_dim`` below :math:`C` restricts :math:`\Lambda` to that rank.
Standardise each channel, for example with training-set statistics, so
that the channels start on an equal footing.

On 17 equal-length datasets of the UEA multivariate archive
(``benchmarks/time_series/run_multivariate.py``: one prototype per class,
``gamma`` equal to the number of channels, the same settings on every
dataset), the three soft-DTW models reach mean test accuracies of 0.710
(:class:`SoftDTWRSLVQ`), 0.709 (:class:`SoftDTWMRSLVQ`) and 0.708
(:class:`SoftDTWLMRSLVQ`), against 0.711 and 0.698 for the 1-nearest
neighbour under independent and dependent DTW. None of the differences
is significant (Friedman test, :math:`p = 0.53`). On these datasets the
learned metric does not raise accuracy; what it adds is the channel
relevances, which show the channels the classes differ in.

Benchmark
---------

On 24 datasets of the UCR time series archive, with the archive's own
train/test splits and nothing tuned:

.. list-table::
   :header-rows: 1

   * - Method
     - Mean accuracy
     - Better / equal / worse than 1NN-DTW
   * - 1-nearest neighbour, Euclidean
     - 0.786
     - 9 / 3 / 12
   * - 1-nearest neighbour, DTW (1NN-DTW)
     - 0.824
     - —
   * - 1-nearest neighbour, DTW with a 10 % band
     - 0.831
     - 8 / 13 / 3
   * - RBF SVM
     - 0.709
     - 5 / 0 / 19
   * - Random forest
     - 0.827
     - 12 / 2 / 10
   * - ``WRSLVQ``
     - 0.815
     - 11 / 1 / 12
   * - ``SoftDTWRSLVQ``, :math:`\gamma = 0.1` (default)
     - 0.862
     - 11 / 5 / 8
   * - ``SoftDTWRSLVQ``, :math:`\gamma = 1`
     - 0.880
     - 16 / 3 / 5
   * - ``SoftDTWRSLVQ``, :math:`\gamma = 0.1`, 2 prototypes per class
     - 0.846
     - 13 / 4 / 7

With :math:`\gamma = 1`, ``SoftDTWRSLVQ`` is more accurate than each of
the six baselines (Wilcoxon signed-rank tests, Holm-adjusted,
:math:`p < 0.05`). With the default :math:`\gamma = 0.1` it has the best
mean rank, but its advantage is significant only over the SVM. The
nearest-neighbour baselines reproduce the error rates published with the
archive. The protocol, the statistics and the accuracy on every dataset
are in :doc:`/benchmarks/index` and ``benchmarks/time_series``.
:doc:`/tutorials/time_series` works through an example.

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
