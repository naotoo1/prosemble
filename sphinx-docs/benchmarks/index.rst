Benchmarks
==========

.. currentmodule:: prosemble.models

How do prototype models compare with standard classifiers? The benchmarks
answer this with a fixed protocol and the statistical tests of Demšar
(2006), and report the results whichever way they fall. The code and full
reports are in the
`benchmarks directory <https://github.com/naotoo1/prosemble/tree/main/benchmarks>`_
of the repository.

How results are compared
------------------------

Every method is run on every dataset, and a dataset's score is one
observation. Methods are ranked on each dataset (rank 1 is best) and the
Friedman test checks whether the mean ranks differ at all. A reference
method is then compared with each other method by a two-sided Wilcoxon
signed-rank test over the datasets, with Holm's correction for the number
of comparisons. Wins, ties and losses count the datasets on which the
reference is better, equal or worse. The statistics are in
:mod:`prosemble.core.evaluation` (see :doc:`/user_guide/model_selection`).

Tabular classification
----------------------

**Protocol.** 36 OpenML datasets, fixed by id; two repeats of stratified
5-fold cross-validation; standardisation fitted on each training split.
Logistic regression, k-nearest neighbours and an RBF SVM are tuned by an
inner grid search; random forest (500 trees) and gradient boosting run at
their defaults. The prosemble models run at one fixed setting on every
dataset (200 iterations, learning rate 0.01), and the soft models are
calibrated with ``calibrate(cv=3)``. The baselines are tuned and the
prosemble models are not, so the comparison favours the baselines.

.. list-table:: Means over 36 datasets, mean ranks in brackets; bold marks the best mean rank
   :header-rows: 1
   :widths: 28 24 24 24

   * - Method
     - Accuracy
     - Log-loss
     - Calibration error
   * - SVM (tuned)
     - **0.885** (3.19)
     - **0.295** (2.67)
     - 0.048 (3.94)
   * - :class:`LGMLVQ`
     - 0.866 (5.12)
     - 3.668 (11.22)
     - 0.283 (11.72)
   * - :class:`DKLMRSLVQ`
     - 0.867 (5.62)
     - 0.329 (3.50)
     - **0.048** (3.92)
   * - Random forest
     - 0.875 (5.75)
     - 0.381 (5.22)
     - 0.081 (6.64)
   * - Logistic regression (tuned)
     - 0.862 (5.96)
     - 0.369 (4.69)
     - 0.060 (5.06)
   * - Gradient boosting
     - 0.875 (5.96)
     - 0.412 (5.83)
     - 0.073 (5.56)
   * - :class:`WLMRSLVQ`
     - 0.866 (6.44)
     - 0.345 (4.72)
     - 0.061 (5.94)
   * - :class:`GMLVQ`
     - 0.855 (6.88)
     - 3.455 (10.78)
     - 0.255 (10.81)
   * - k-nearest neighbours (tuned)
     - 0.864 (6.92)
     - 1.924 (9.56)
     - 0.085 (6.36)
   * - :class:`WRSLVQ`
     - 0.853 (7.54)
     - 0.364 (5.22)
     - 0.047 (4.19)
   * - :class:`DKRSLVQ`
     - 0.856 (8.06)
     - 0.358 (4.72)
     - 0.049 (4.42)
   * - :class:`GLVQ`
     - 0.806 (10.56)
     - 2.432 (9.86)
     - 0.168 (9.44)

**Findings**, with :class:`DKLMRSLVQ` as the reference (Holm-adjusted
Wilcoxon tests, :math:`p < 0.05`):

* **Accuracy.** The tuned SVM is more accurate than DKLMRSLVQ (7 wins, 1
  tie, 28 losses). DKLMRSLVQ is statistically level with random forest,
  gradient boosting, logistic regression, kNN and LGMLVQ, and more accurate
  than WRSLVQ, DKRSLVQ and GLVQ.
* **Log-loss.** The SVM is better; DKLMRSLVQ is second, and better than
  gradient boosting, WRSLVQ and DKRSLVQ.
* **Calibration.** DKLMRSLVQ has the lowest mean rank, level with the SVM,
  and is better calibrated than random forest, gradient boosting, kNN and
  WLMRSLVQ.
* GLVQ, GMLVQ and LGMLVQ are not probabilistic models: their
  ``predict_proba`` scores have log-losses of 2.4 to 3.7. Use the soft
  models when probabilities matter (:doc:`/user_guide/probabilities`).

The full report, with balanced accuracy, accuracy on the 80 % most
confident samples and the per-dataset scores, is
`benchmarks/classification/RESULTS.md
<https://github.com/naotoo1/prosemble/blob/main/benchmarks/classification/RESULTS.md>`_.

Time-series classification
--------------------------

**Protocol.** 24 univariate datasets of the UCR archive (2018 release),
with the archive's own train/test splits; every series is z-normalised.
The baselines are 1-nearest neighbour under the Euclidean distance, under
DTW and under DTW with a 10 % band; an RBF SVM and a random forest at
their defaults; and :class:`WRSLVQ`. :class:`SoftDTWRSLVQ` runs at fixed
settings (150 iterations, learning rate 0.05) in three configurations,
all fixed before the run. Nothing is tuned.

.. list-table:: Test accuracy over 24 datasets
   :header-rows: 1
   :widths: 46 18 36

   * - Method
     - Mean accuracy
     - Better / equal / worse than 1NN-DTW
   * - 1NN, Euclidean
     - 0.786
     - 9 / 3 / 12
   * - 1NN-DTW
     - 0.824
     - —
   * - 1NN-DTW, 10 % band
     - 0.831
     - 8 / 13 / 3
   * - RBF SVM
     - 0.709
     - 5 / 0 / 19
   * - Random forest
     - 0.827
     - 12 / 2 / 10
   * - :class:`WRSLVQ`
     - 0.815
     - 11 / 1 / 12
   * - :class:`SoftDTWRSLVQ`, :math:`\gamma = 0.1` (default)
     - 0.862
     - 11 / 5 / 8
   * - :class:`SoftDTWRSLVQ`, :math:`\gamma = 1`
     - **0.880**
     - 16 / 3 / 5
   * - :class:`SoftDTWRSLVQ`, :math:`\gamma = 0.1`, 2 prototypes per class
     - 0.846
     - 13 / 4 / 7

**Findings.** Each configuration was compared with the six baselines
separately (Holm-adjusted Wilcoxon tests, :math:`p < 0.05`):

* With :math:`\gamma = 1`, SoftDTWRSLVQ is more accurate than each of the
  six baselines, and has the best mean rank (2.60 of 7).
* With the default :math:`\gamma = 0.1` it also has the best mean rank, but
  is significantly better only than the SVM.
* :math:`\gamma = 1` is the best of three configurations evaluated on the
  same test sets, which flatters it somewhat; all three are reported.
* The nearest-neighbour baselines reproduce the error rates published with
  the archive on all 24 datasets.
* SoftDTWRSLVQ is much slower to train: about three hours of computation
  over the 24 datasets for one configuration, against nine minutes for
  1NN-DTW.

The full report is
`benchmarks/time_series/RESULTS.md
<https://github.com/naotoo1/prosemble/blob/main/benchmarks/time_series/RESULTS.md>`_.

Reproducing the results
-----------------------

.. code-block:: bash

   python benchmarks/classification/run.py --out results.jsonl
   python benchmarks/classification/report.py results.jsonl \
       --reference DKLMRSLVQ > RESULTS.md

   python benchmarks/time_series/run.py --out ts_results.jsonl
   python benchmarks/time_series/run.py --summary ts_results.jsonl

Runs are resumable and can be split into shards that run in parallel; see
the README of each benchmark.

.. rubric:: References

* Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple
  Data Sets. *Journal of Machine Learning Research*, 7, 1–30.
* Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65–70.
