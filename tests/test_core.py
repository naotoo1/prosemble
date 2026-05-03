"""
Prosemble core distance functions test suite.

Tests cover:
- Manhattan distance
- Euclidean distance
- Squared Euclidean distance
- Lp-norm distance (p=1, p=2)
- Omega distance (projection-based)
- Local omega distance (multiple projections)

All tests verify correctness against scikit-learn pairwise distances.
"""

import pytest
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import prosemble as ps


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(32, 6)
    Y = np.random.randn(8, 6)
    return X, Y


class TestManhattanDistance:
    """Test Manhattan (L1) distance implementation."""

    def test_correctness_against_sklearn(self, sample_data):
        """Verify Manhattan distance matches sklearn."""
        X, Y = sample_data
        nx, ny = len(X), len(Y)

        # Compute distances
        actual = np.array([
            [ps.core.manhattan_distance(X[i], Y[j]) for j in range(ny)]
            for i in range(nx)
        ])

        expected = np.array([
            [pairwise_distances(
                X[i].reshape(1, -1),
                Y[j].reshape(1, -1),
                metric="manhattan"
            )[0, 0] for j in range(ny)]
            for i in range(nx)
        ])

        np.testing.assert_array_almost_equal(actual, expected, decimal=2)


class TestEuclideanDistance:
    """Test Euclidean (L2) distance implementation."""

    def test_correctness_against_sklearn(self, sample_data):
        """Verify Euclidean distance matches sklearn."""
        X, Y = sample_data
        nx, ny = len(X), len(Y)

        actual = np.array([
            [ps.core.euclidean_distance(X[i], Y[j]) for j in range(ny)]
            for i in range(nx)
        ])

        expected = np.array([
            [pairwise_distances(
                X[i].reshape(1, -1),
                Y[j].reshape(1, -1),
                metric="euclidean"
            )[0, 0] for j in range(ny)]
            for i in range(nx)
        ])

        np.testing.assert_array_almost_equal(actual, expected, decimal=3)


class TestSquaredEuclideanDistance:
    """Test squared Euclidean distance implementation."""

    def test_correctness_against_sklearn(self, sample_data):
        """Verify squared Euclidean distance matches sklearn."""
        X, Y = sample_data
        nx, ny = len(X), len(Y)

        actual = np.array([
            [ps.core.squared_euclidean_distance(X[i], Y[j]) for j in range(ny)]
            for i in range(nx)
        ])

        expected = np.array([
            [pairwise_distances(
                X[i].reshape(1, -1),
                Y[j].reshape(1, -1),
                metric="sqeuclidean"
            )[0, 0] for j in range(ny)]
            for i in range(nx)
        ])

        np.testing.assert_array_almost_equal(actual, expected, decimal=2)


class TestLpNormDistance:
    """Test Lp-norm distance implementation."""

    @pytest.mark.parametrize("p,metric", [
        (1, "l1"),
        (2, "l2"),
    ])
    def test_correctness_against_sklearn(self, sample_data, p, metric):
        """Verify Lp-norm distance matches sklearn for p=1,2."""
        X, Y = sample_data
        nx, ny = len(X), len(Y)

        actual = np.array([
            [ps.core.lpnorm_distance(X[i], Y[j], p) for j in range(ny)]
            for i in range(nx)
        ])

        expected = np.array([
            [pairwise_distances(
                X[i].reshape(1, -1),
                Y[j].reshape(1, -1),
                metric=metric
            )[0, 0] for j in range(ny)]
            for i in range(nx)
        ])

        np.testing.assert_array_almost_equal(actual, expected, decimal=2)


class TestOmegaDistance:
    """Test Omega (projection-based) distance."""

    def test_identity_omega_equals_squared_euclidean(self, sample_data):
        """Verify identity projection gives squared Euclidean distance."""
        X, Y = sample_data
        nx, ny, d = len(X), len(Y), X.shape[1]

        omega = np.eye(d, d)

        actual = np.array([
            [ps.core.omega_distance(X[i], Y[j], omega) for j in range(ny)]
            for i in range(nx)
        ])

        expected = np.array([
            [pairwise_distances(
                X[i].reshape(1, -1),
                Y[j].reshape(1, -1),
                metric="l2"
            )[0, 0] ** 2 for j in range(ny)]
            for i in range(nx)
        ])

        np.testing.assert_array_almost_equal(actual, expected, decimal=2)


class TestLomegaDistance:
    """Test local omega (multiple projection matrices) distance."""

    def test_identity_omegas_equals_squared_euclidean(self, sample_data):
        """Verify identity projections give squared Euclidean distances."""
        X, Y = sample_data
        d = X.shape[1]
        ny = len(Y)

        # Stack identity matrices
        omega = np.eye(d, d)
        omegas = np.stack([omega for _ in range(ny)], axis=0)

        actual = ps.core.lomega_distance(X, Y, omegas)

        expected = np.array([
            [pairwise_distances(
                X[i].reshape(1, -1),
                Y[j].reshape(1, -1),
                metric="l2"
            )[0, 0] ** 2 for j in range(ny)]
            for i in range(len(X))
        ])

        np.testing.assert_array_almost_equal(actual, expected, decimal=1)


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_zero_vectors(self):
        """Test distance between zero vectors."""
        x = np.zeros(5)
        y = np.zeros(5)

        assert ps.core.euclidean_distance(x, y) == 0.0
        assert ps.core.manhattan_distance(x, y) == 0.0
        assert ps.core.squared_euclidean_distance(x, y) == 0.0

    def test_identical_vectors(self):
        """Test distance from vector to itself."""
        x = np.random.randn(10)

        assert ps.core.euclidean_distance(x, x) == pytest.approx(0.0, abs=1e-10)
        assert ps.core.manhattan_distance(x, x) == pytest.approx(0.0, abs=1e-10)
        assert ps.core.squared_euclidean_distance(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_large_values(self):
        """Test with large coordinate values."""
        x = np.array([1e6, 2e6, 3e6])
        y = np.array([0, 0, 0])

        d = ps.core.euclidean_distance(x, y)
        assert np.isfinite(d)
        assert d > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
