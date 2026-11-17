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
dataset), the two soft-DTW models reach mean test accuracies of 0.710
(:class:`SoftDTWRSLVQ`) and 0.709 (:class:`SoftDTWMRSLVQ`), against
0.711 and 0.698 for the 1-nearest neighbour under independent and
dependent DTW. None of the differences is significant (Friedman test,
:math:`p = 0.50`). On these datasets the
learned metric does not raise accuracy; what it adds is the channel
relevances, which show the channels the classes differ in.

One class of series
-------------------

:class:`OCAlignmentDensity` models the series of one class, for telling
normal series from anomalous ones. It is fitted to normal series only and
gives each series its log density under the model.

A series :math:`x \in \mathbb{R}^{T \times C}` is generated from a
prototype series :math:`w_k` by a monotone alignment and Gaussian noise:

.. math::

   p(x) = \sum_k \pi_k \sum_a \frac{\tau(a)}{Z_\tau}
          \prod_{t=1}^{T} \mathcal{N}\big(x_t;\, w_{k, a_t},\, \Sigma_{k, a_t}\big).

The alignment starts at the first prototype position, ends at the last,
and advances by 0, 1 or 2 positions per time step with probabilities
:math:`\tau`; :math:`Z_\tau` normalises them over the alignments that end
at the last position. Each :math:`x_t` is emitted exactly once from a
normalised Gaussian, so :math:`p` is a proper density, and
:math:`\log p(x)` is a forward recursion like soft-DTW's. Expectation–
maximisation fits the prototypes, the variances, the weights
:math:`\pi_k` and :math:`\tau`.

The prototypes start at soft-DTW medoids of the normal series (three by
default). Each prototype position has its own variance per channel,
shrunk towards the variance of the channel with a strength chosen by
empirical Bayes, so that positions which few time points reach still get
a sensible variance; ``covariance='shared'`` uses one full covariance over
the channels instead.

.. code-block:: python

   from prosemble.models import OCAlignmentDensity

   # X_normal: normal series only, one per row (T x C values, time-major)
   model = OCAlignmentDensity(n_channels=C, nu=0.05)
   model.fit(X_normal)

   model.score_samples(X_new)            # log density; larger is more normal
   model.decision_function(X_new)        # in (0, 1), boundary at 0.5
   model.predict(X_new)                  # 1 normal, -1 anomalous
   model.prototypes_                     # (3, T, C) prototype series
   model.step_probabilities_             # probabilities of steps 0, 1, 2

The boundary of :meth:`~OCAlignmentDensity.decision_function` is placed
so that the fraction ``nu`` of the training series falls outside it. An
EM step costs time proportional to the number of training series times
the number of prototypes times :math:`T^2`.

On the one-class benchmark (``benchmarks/time_series/run_one_class.py``),
each class of a dataset in turn is the normal class, and the AUROC for
telling its test series from the others is averaged over the classes. On
the 14 UCR datasets not used for design choices:

.. list-table::
   :header-rows: 1

   * - Method
     - Mean AUROC
     - Better / worse than :class:`OCAlignmentDensity`
   * - :class:`OCAlignmentDensity`
     - 0.787
     -
   * - Nearest normal series, DTW
     - 0.745
     - 3 / 11
   * - Nearest normal series, Euclidean
     - 0.741
     - 4 / 10
   * - One-class SVM
     - 0.718
     - 2 / 12
   * - Isolation forest
     - 0.706
     - 1 / 13

The difference to the nearest-neighbour detector under DTW is significant
(Holm-adjusted Wilcoxon test, :math:`p = 0.041`). On the 13 UEA
multivariate datasets not used for design choices it is second, at 0.708
against 0.716 for the nearest-neighbour detector under DTW (not
significant), and ahead of the others. With few training series per
class, results depend on the medoids that start EM.

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
