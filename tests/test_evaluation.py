"""Statistics for comparing classifiers over several datasets."""

import numpy as np
import pytest
from scipy import stats

from prosemble.core.evaluation import (
    compare_to_reference, expected_calibration_error, friedman_test,
    holm_correction, nemenyi_critical_difference, rank_methods,
    selective_accuracy,
)


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.6, 0.95, size=(20, 1))
    # Method 0 is best, method 3 worst, with noise.
    return base + np.array([0.04, 0.02, 0.0, -0.03]) + rng.normal(0, 0.01, size=(20, 4))


class TestRanks:

    def test_best_gets_rank_one_and_ties_share(self):
        ranks = rank_methods([[0.9, 0.8, 0.8], [0.5, 0.7, 0.6]])
        np.testing.assert_array_equal(ranks, [[1.0, 2.5, 2.5], [3.0, 1.0, 2.0]])

    def test_losses_rank_the_smallest_first(self):
        np.testing.assert_array_equal(
            rank_methods([[0.2, 0.5], [0.9, 0.1]], higher_is_better=False), [[1, 2], [2, 1]])

    @pytest.mark.parametrize('bad', [[[0.5, 0.6]], [[0.5], [0.6]], [[0.5, np.nan], [0.1, 0.2]]])
    def test_invalid_tables(self, bad):
        with pytest.raises(ValueError):
            rank_methods(bad)


class TestFriedman:

    def test_matches_scipy_without_ties(self, table):
        result = friedman_test(table)
        reference = stats.friedmanchisquare(*table.T)
        assert result.statistic == pytest.approx(reference.statistic, rel=1e-10)
        assert result.p_value == pytest.approx(reference.pvalue, rel=1e-8)

    def test_iman_davenport_formula(self, table):
        result = friedman_test(table)
        n, k = table.shape
        f = (n - 1) * result.statistic / (n * (k - 1) - result.statistic)
        assert result.iman_davenport == pytest.approx(f)
        assert result.iman_davenport_p_value == pytest.approx(stats.f.sf(f, k - 1, (k - 1) * (n - 1)))

    def test_detects_a_consistent_ordering_and_not_noise(self, table):
        assert friedman_test(table).p_value < 1e-6
        assert np.all(np.diff(friedman_test(table).mean_ranks) > 0)
        noise = np.random.default_rng(1).uniform(size=(20, 4))
        assert friedman_test(noise).p_value > 0.01

    def test_perfect_agreement(self):
        scores = np.tile([0.9, 0.8, 0.7], (6, 1))
        result = friedman_test(scores)
        np.testing.assert_array_equal(result.mean_ranks, [1, 2, 3])
        assert result.iman_davenport == np.inf and result.iman_davenport_p_value == 0.0


class TestNemenyi:

    @pytest.mark.parametrize('k, q', [(2, 1.960), (3, 2.343), (4, 2.569), (5, 2.728),
                                      (6, 2.850), (8, 3.031), (10, 3.164)])
    def test_critical_values_of_demsar_2006(self, k, q):
        """Table 5(a) of Demšar (2006): q at alpha = 0.05."""
        n = 14
        cd = nemenyi_critical_difference(k, n)
        assert cd == pytest.approx(q * np.sqrt(k * (k + 1) / (6 * n)), abs=2e-3)

    def test_shrinks_with_more_datasets(self):
        assert nemenyi_critical_difference(5, 40) < nemenyi_critical_difference(5, 10)

    def test_invalid(self):
        with pytest.raises(ValueError):
            nemenyi_critical_difference(1, 10)


class TestHolm:

    def test_known_example(self):
        np.testing.assert_allclose(holm_correction([0.01, 0.04, 0.03, 0.005]),
                                   [0.03, 0.06, 0.06, 0.02])

    def test_is_monotone_and_capped(self):
        adjusted = holm_correction([0.5, 0.9, 0.2])
        assert np.all(adjusted <= 1.0)
        order = np.argsort([0.5, 0.9, 0.2])
        assert np.all(np.diff(adjusted[order]) >= 0)


class TestCompareToReference:

    def test_orientation_counts_and_correction(self, table):
        results = compare_to_reference(table, reference=0)
        assert [r.method for r in results] == [1, 2, 3]
        for r in results:
            assert r.mean_difference > 0
            assert r.wins + r.ties + r.losses == 20
            assert r.holm_p_value >= r.p_value
        assert results[-1].holm_p_value < 0.001

    def test_matches_scipy_wilcoxon(self, table):
        r = compare_to_reference(table, reference=1)[0]
        expected = stats.wilcoxon(table[:, 1] - table[:, 0], zero_method='pratt').pvalue
        assert r.p_value == pytest.approx(expected)

    def test_losses(self, table):
        results = compare_to_reference(1.0 - table, reference=0, higher_is_better=False)
        assert all(r.mean_difference > 0 for r in results)

    def test_identical_methods(self):
        scores = np.tile([[0.8, 0.8]], (5, 1))
        r = compare_to_reference(scores, reference=0)[0]
        assert (r.p_value, r.ties) == (1.0, 5)


class TestProbabilityMeasures:

    def test_ece_is_zero_for_calibrated_and_large_for_overconfident(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.5, 1.0, size=20000)
        y = (rng.uniform(size=20000) < p).astype(int)        # class 1 correct with probability p
        proba = np.stack([1 - p, p], axis=1)
        assert expected_calibration_error(proba, np.where(y == 1, 1, 0)) < 0.02
        sharp = np.stack([np.zeros_like(p), np.ones_like(p)], axis=1)
        assert expected_calibration_error(sharp, y) == pytest.approx(1 - y.mean(), abs=1e-9)

    def test_selective_accuracy(self):
        proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.55, 0.45]])
        y = np.array([0, 1, 1, 1])
        assert selective_accuracy(proba, y, 1.0) == pytest.approx(0.5)
        assert selective_accuracy(proba, y, 0.5) == pytest.approx(1.0)
        with pytest.raises(ValueError):
            selective_accuracy(proba, y, 0.0)
