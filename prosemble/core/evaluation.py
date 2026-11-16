"""
Statistics for comparing classifiers over several datasets.

A score table holds one row per dataset and one column per method (for
example the mean accuracy over cross-validation folds). The functions
here follow the protocol of Demšar (2006): rank the methods on each
dataset, test with the Friedman test whether the mean ranks differ,
compare all methods with the Nemenyi critical difference, and compare
one method with each of the others by Wilcoxon signed-rank tests with
Holm's correction for multiple testing.

They also provide two measures of the probabilities a classifier
returns: the expected calibration error and the accuracy when the
classifier may abstain on its least confident samples.

References
----------
.. [1] Demšar, J. (2006). Statistical Comparisons of Classifiers over
       Multiple Data Sets. Journal of Machine Learning Research, 7, 1-30.
.. [2] Holm, S. (1979). A Simple Sequentially Rejective Multiple Test
       Procedure. Scandinavian Journal of Statistics, 6(2), 65-70.
"""

from typing import NamedTuple

import numpy as np
from scipy import stats


class FriedmanResult(NamedTuple):
    """Friedman test of equal mean ranks.

    Attributes
    ----------
    statistic : float
        Friedman's chi-squared statistic.
    p_value : float
        p-value of the chi-squared approximation.
    iman_davenport : float
        Iman and Davenport's F statistic, a less conservative version.
    iman_davenport_p_value : float
        p-value of the F statistic.
    mean_ranks : ndarray of shape (n_methods,)
        Mean rank of each method; rank 1 is the best.
    """

    statistic: float
    p_value: float
    iman_davenport: float
    iman_davenport_p_value: float
    mean_ranks: np.ndarray


class PairwiseResult(NamedTuple):
    """Wilcoxon signed-rank test of one method against another.

    Attributes
    ----------
    method : int
        Column of the method compared with the reference.
    mean_difference : float
        Mean of reference score minus method score over the datasets.
    wins, ties, losses : int
        Datasets on which the reference scores higher, equal, lower.
    p_value : float
        Two-sided p-value of the Wilcoxon signed-rank test.
    holm_p_value : float
        p-value adjusted by Holm's procedure over all comparisons.
    """

    method: int
    mean_difference: float
    wins: int
    ties: int
    losses: int
    p_value: float
    holm_p_value: float


def _table(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] < 2:
        raise ValueError("scores must have shape (n_datasets >= 2, n_methods >= 2)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    return scores


def rank_methods(scores, higher_is_better=True):
    """Rank the methods on each dataset; tied methods share the mean rank.

    Parameters
    ----------
    scores : array-like of shape (n_datasets, n_methods)
    higher_is_better : bool
        True for accuracy-like scores, False for losses.

    Returns
    -------
    ranks : ndarray of shape (n_datasets, n_methods)
        Rank 1 is the best method on that dataset.
    """
    scores = _table(scores)
    oriented = -scores if higher_is_better else scores
    return np.apply_along_axis(stats.rankdata, 1, oriented)


def friedman_test(scores, higher_is_better=True):
    """Friedman test that all methods have the same mean rank.

    Parameters
    ----------
    scores : array-like of shape (n_datasets, n_methods)
    higher_is_better : bool

    Returns
    -------
    FriedmanResult
    """
    ranks = rank_methods(scores, higher_is_better)
    n, k = ranks.shape
    mean_ranks = ranks.mean(axis=0)
    chi2 = 12.0 * n / (k * (k + 1)) * (np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4.0)
    p_chi2 = float(stats.chi2.sf(chi2, k - 1))
    if chi2 >= n * (k - 1):
        f_stat, p_f = np.inf, 0.0
    else:
        f_stat = (n - 1) * chi2 / (n * (k - 1) - chi2)
        p_f = float(stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1)))
    return FriedmanResult(float(chi2), p_chi2, float(f_stat), p_f, mean_ranks)


def nemenyi_critical_difference(n_methods, n_datasets, alpha=0.05):
    """Smallest difference of mean ranks that the Nemenyi test calls significant.

    .. math::

        CD = q_\\alpha \\sqrt{\\frac{k (k + 1)}{6 N}},

    with :math:`q_\\alpha` the studentized range quantile divided by
    :math:`\\sqrt 2`.

    Parameters
    ----------
    n_methods : int
    n_datasets : int
    alpha : float

    Returns
    -------
    cd : float
    """
    if n_methods < 2 or n_datasets < 1:
        raise ValueError("need at least two methods and one dataset")
    q = stats.studentized_range.ppf(1.0 - alpha, n_methods, np.inf) / np.sqrt(2.0)
    return float(q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets)))


def holm_correction(p_values):
    """Holm's step-down adjustment of a family of p-values.

    Parameters
    ----------
    p_values : array-like of shape (m,)

    Returns
    -------
    adjusted : ndarray of shape (m,)
        Reject hypothesis ``i`` at level alpha when ``adjusted[i] <= alpha``.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (m - position) * p[index]))
        adjusted[index] = running
    return adjusted


def compare_to_reference(scores, reference, higher_is_better=True):
    """Wilcoxon signed-rank tests of one method against each of the others.

    Parameters
    ----------
    scores : array-like of shape (n_datasets, n_methods)
    reference : int
        Column of the method compared with all others.
    higher_is_better : bool

    Returns
    -------
    results : list of PairwiseResult
        One per other method, in column order. ``mean_difference`` and
        wins are oriented so that positive means the reference is better.
    """
    scores = _table(scores)
    sign = 1.0 if higher_is_better else -1.0
    others = [j for j in range(scores.shape[1]) if j != reference]
    rows = []
    for j in others:
        diff = sign * (scores[:, reference] - scores[:, j])
        if np.all(diff == 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(diff, zero_method='pratt', alternative='two-sided').pvalue)
        rows.append((j, float(diff.mean()), int(np.sum(diff > 0)), int(np.sum(diff == 0)),
                     int(np.sum(diff < 0)), p))
    adjusted = holm_correction([r[5] for r in rows])
    return [PairwiseResult(*r, float(a)) for r, a in zip(rows, adjusted)]


def expected_calibration_error(proba, y, n_bins=10):
    """Expected calibration error of the top-class probability.

    Samples are binned by the probability of the predicted class; the
    error is the bin-size-weighted mean absolute difference between the
    accuracy and the mean probability of each bin.

    Parameters
    ----------
    proba : array-like of shape (n_samples, n_classes)
    y : array-like of shape (n_samples,)
        Column indices of the true classes.
    n_bins : int

    Returns
    -------
    ece : float
    """
    proba = np.asarray(proba, dtype=np.float64)
    y = np.asarray(y)
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(np.float64)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        inside = bins == b
        if np.any(inside):
            ece += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(ece)


def selective_accuracy(proba, y, coverage):
    """Accuracy on the ``coverage`` fraction of samples with the highest confidence.

    Parameters
    ----------
    proba : array-like of shape (n_samples, n_classes)
    y : array-like of shape (n_samples,)
        Column indices of the true classes.
    coverage : float
        Fraction in (0, 1] of samples kept; the others are abstained on.

    Returns
    -------
    accuracy : float
    """
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage!r}")
    proba = np.asarray(proba, dtype=np.float64)
    y = np.asarray(y)
    kept = max(1, int(round(coverage * len(y))))
    order = np.argsort(-proba.max(axis=1), kind='stable')[:kept]
    return float(np.mean(proba[order].argmax(axis=1) == y[order]))
