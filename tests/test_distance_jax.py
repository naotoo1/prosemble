"""
Tests for JAX distance functions.

This test suite validates:
1. Correctness: JAX vs NumPy parity
2. Numerical stability: Edge cases
3. Shape compatibility
4. Performance: Basic benchmarks
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from prosemble.core.distance_jax import (
        euclidean_distance_matrix,
        squared_euclidean_distance_matrix,
        manhattan_distance_matrix,
        lpnorm_distance_matrix,
        omega_distance_matrix,
        lomega_distance_matrix,
        gaussian_kernel_matrix,
        polynomial_kernel_matrix,
        euclidean_distance,
        squared_euclidean_distance,
        estimate_sigma,
        safe_divide,
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from prosemble.core.distance import (
    euclidean_distance as np_euclidean_distance,
    squared_euclidean_distance as np_squared_euclidean_distance,
    manhattan_distance as np_manhattan_distance,
    lpnorm_distance as np_lpnorm_distance,
    omega_distance as np_omega_distance,
    lomega_distance as np_lomega_distance,
)

# Skip all tests if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


class TestEuclideanDistance:
    """Test Euclidean distance computations."""

    def test_matrix_shape(self):
        """Test output shape is correct."""
        X = jnp.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
        Y = jnp.array([[0, 0], [1, 1]])  # (2, 2)

        D = euclidean_distance_matrix(X, Y)

        assert D.shape == (3, 2)

    def test_self_distance_zero(self):
        """Test that distance from point to itself is zero."""
        X = jnp.array([[1, 2, 3], [4, 5, 6]])

        D = euclidean_distance_matrix(X, X)

        np.testing.assert_allclose(jnp.diag(D), 0.0, atol=1e-6)

    def test_known_values(self):
        """Test against known distance values."""
        X = jnp.array([[0, 0]])
        Y = jnp.array([[3, 4]])  # 3-4-5 triangle

        D = euclidean_distance_matrix(X, Y)

        assert np.isclose(D[0, 0], 5.0, atol=1e-6)

    def test_symmetry(self):
        """Test that D(X, Y) = D(Y, X).T"""
        np.random.seed(42)
        X = jnp.array(np.random.randn(5, 3))
        Y = jnp.array(np.random.randn(4, 3))

        D_xy = euclidean_distance_matrix(X, Y)
        D_yx = euclidean_distance_matrix(Y, X)

        np.testing.assert_allclose(D_xy, D_yx.T, rtol=1e-5)

    def test_pairwise_comparison(self):
        """Compare matrix function with pairwise function."""
        X = jnp.array([[1, 2], [3, 4]])
        Y = jnp.array([[0, 0], [5, 5]])

        # Matrix version
        D_matrix = euclidean_distance_matrix(X, Y)

        # Pairwise version
        D_pairwise = jnp.array([
            [euclidean_distance(X[i], Y[j]) for j in range(len(Y))]
            for i in range(len(X))
        ])

        np.testing.assert_allclose(D_matrix, D_pairwise, rtol=1e-5)

    def test_numpy_parity(self):
        """Test that JAX and NumPy implementations agree."""
        np.random.seed(42)
        X_np = np.random.randn(10, 5)
        Y_np = np.random.randn(8, 5)

        X_jax = jnp.array(X_np)
        Y_jax = jnp.array(Y_np)

        # Compute with JAX
        D_jax = euclidean_distance_matrix(X_jax, Y_jax)

        # Compute with NumPy (element-wise)
        D_numpy = np.array([
            [np_euclidean_distance(X_np[i], Y_np[j]) for j in range(len(Y_np))]
            for i in range(len(X_np))
        ])

        np.testing.assert_allclose(D_jax, D_numpy, rtol=1e-5, atol=1e-7)


class TestSquaredEuclideanDistance:
    """Test squared Euclidean distance computations."""

    def test_squared_vs_euclidean(self):
        """Test that squared distance = distance²."""
        X = jnp.array([[1, 2, 3], [4, 5, 6]])
        Y = jnp.array([[0, 0, 0], [1, 1, 1]])

        D = euclidean_distance_matrix(X, Y)
        D_sq = squared_euclidean_distance_matrix(X, Y)

        np.testing.assert_allclose(D ** 2, D_sq, rtol=1e-5)

    def test_non_negative(self):
        """Test that all squared distances are non-negative."""
        np.random.seed(42)
        X = jnp.array(np.random.randn(20, 10))
        Y = jnp.array(np.random.randn(15, 10))

        D_sq = squared_euclidean_distance_matrix(X, Y)

        assert jnp.all(D_sq >= 0)

    def test_numpy_parity(self):
        """Test parity with NumPy implementation."""
        np.random.seed(42)
        X_np = np.random.randn(10, 5)
        Y_np = np.random.randn(8, 5)

        X_jax = jnp.array(X_np)
        Y_jax = jnp.array(Y_np)

        # JAX version
        D_sq_jax = squared_euclidean_distance_matrix(X_jax, Y_jax)

        # NumPy version
        D_sq_numpy = np.array([
            [np_squared_euclidean_distance(X_np[i], Y_np[j])
             for j in range(len(Y_np))]
            for i in range(len(X_np))
        ])

        np.testing.assert_allclose(D_sq_jax, D_sq_numpy, rtol=1e-5, atol=1e-7)


class TestManhattanDistance:
    """Test Manhattan distance computations."""

    def test_known_values(self):
        """Test against known Manhattan distance."""
        X = jnp.array([[0, 0]])
        Y = jnp.array([[3, 4]])

        D = manhattan_distance_matrix(X, Y)

        assert np.isclose(D[0, 0], 7.0)  # |3-0| + |4-0| = 7

    def test_numpy_parity(self):
        """Test parity with NumPy implementation."""
        np.random.seed(42)
        X_np = np.random.randn(10, 5)
        Y_np = np.random.randn(8, 5)

        X_jax = jnp.array(X_np)
        Y_jax = jnp.array(Y_np)

        # JAX version
        D_jax = manhattan_distance_matrix(X_jax, Y_jax)

        # NumPy version (using loop - original implementation)
        D_numpy = np.array([
            [np_manhattan_distance(X_np[i], Y_np[j])
             for j in range(len(Y_np))]
            for i in range(len(X_np))
        ])

        np.testing.assert_allclose(D_jax, D_numpy, rtol=1e-5, atol=1e-7)

    def test_unit_hypercube(self):
        """Test on unit hypercube corners."""
        X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        D = manhattan_distance_matrix(X, X)

        # Diagonal should be zero
        np.testing.assert_allclose(jnp.diag(D), 0.0, atol=1e-6)

        # Adjacent corners should have distance 1
        assert np.isclose(D[0, 1], 1.0)
        assert np.isclose(D[0, 2], 1.0)

        # Opposite corners should have distance 2
        assert np.isclose(D[0, 3], 2.0)


class TestLpNormDistance:
    """Test L-p norm distance computations."""

    def test_l1_equals_manhattan(self):
        """Test that L1 norm equals Manhattan distance."""
        X = jnp.array([[1, 2, 3], [4, 5, 6]])
        Y = jnp.array([[0, 0, 0], [1, 1, 1]])

        D_l1 = lpnorm_distance_matrix(X, Y, p=1)
        D_manhattan = manhattan_distance_matrix(X, Y)

        np.testing.assert_allclose(D_l1, D_manhattan, rtol=1e-5)

    def test_l2_equals_euclidean(self):
        """Test that L2 norm equals Euclidean distance."""
        X = jnp.array([[1, 2, 3], [4, 5, 6]])
        Y = jnp.array([[0, 0, 0], [1, 1, 1]])

        D_l2 = lpnorm_distance_matrix(X, Y, p=2)
        D_euclidean = euclidean_distance_matrix(X, Y)

        np.testing.assert_allclose(D_l2, D_euclidean, rtol=1e-5)

    def test_linf_is_max(self):
        """Test that L-infinity norm is max absolute difference."""
        X = jnp.array([[0, 0, 0]])
        Y = jnp.array([[1, 5, 3]])  # max diff is 5

        D = lpnorm_distance_matrix(X, Y, p=jnp.inf)

        assert np.isclose(D[0, 0], 5.0)

    def test_numpy_parity(self):
        """Test parity with NumPy implementation."""
        np.random.seed(42)
        X_np = np.random.randn(10, 5)
        Y_np = np.random.randn(8, 5)

        X_jax = jnp.array(X_np)
        Y_jax = jnp.array(Y_np)

        for p in [1, 2, 3, np.inf]:
            D_jax = lpnorm_distance_matrix(X_jax, Y_jax, p=p)

            D_numpy = np.array([
                [np_lpnorm_distance(X_np[i], Y_np[j], p=p)
                 for j in range(len(Y_np))]
                for i in range(len(X_np))
            ])

            np.testing.assert_allclose(D_jax, D_numpy, rtol=1e-5, atol=1e-7)


class TestOmegaDistance:
    """Test projected distance with Omega matrix."""

    def test_identity_omega(self):
        """Test that identity projection gives Euclidean distance."""
        X = jnp.array([[1, 2, 3], [4, 5, 6]])
        Y = jnp.array([[0, 0, 0], [1, 1, 1]])
        omega = jnp.eye(3)  # Identity

        D_omega = omega_distance_matrix(X, Y, omega)
        D_euclidean_sq = squared_euclidean_distance_matrix(X, Y)

        np.testing.assert_allclose(D_omega, D_euclidean_sq, rtol=1e-5)

    def test_projection_reduces_dimension(self):
        """Test projection to lower dimension."""
        X = jnp.array([[1, 2, 3], [4, 5, 6]])
        Y = jnp.array([[0, 0, 0]])

        # Project to first 2 dimensions only
        omega = jnp.array([[1, 0], [0, 1], [0, 0]], dtype=jnp.float32)

        D = omega_distance_matrix(X, Y, omega)

        # Manually compute: project and compute distance
        X_proj = X @ omega  # [[1, 2], [4, 5]]
        Y_proj = Y @ omega  # [[0, 0]]

        expected = squared_euclidean_distance_matrix(X_proj, Y_proj)

        np.testing.assert_allclose(D, expected, rtol=1e-5)

    def test_numpy_parity(self):
        """Test parity with NumPy implementation."""
        np.random.seed(42)
        X_np = np.random.randn(10, 5)
        Y_np = np.random.randn(8, 5)
        omega_np = np.random.randn(5, 3)

        X_jax = jnp.array(X_np)
        Y_jax = jnp.array(Y_np)
        omega_jax = jnp.array(omega_np)

        # JAX version
        D_jax = omega_distance_matrix(X_jax, Y_jax, omega_jax)

        # NumPy version
        D_numpy = np.array([
            [np_omega_distance(X_np[i], Y_np[j], omega_np)
             for j in range(len(Y_np))]
            for i in range(len(X_np))
        ])

        np.testing.assert_allclose(D_jax, D_numpy, rtol=1e-4, atol=1e-6)


class TestLomegaDistance:
    """Test local omega (multiple projection matrices) distance."""

    def test_shape(self):
        """Test output shape is correct."""
        X = jnp.array(np.random.randn(10, 5))
        Y = jnp.array(np.random.randn(3, 5))
        omegas = jnp.array(np.random.randn(3, 5, 2))  # 3 prototypes, each with (5,2) omega

        D = lomega_distance_matrix(X, Y, omegas)

        assert D.shape == (10, 3)

    def test_numpy_parity(self):
        """Test parity with NumPy implementation."""
        np.random.seed(42)
        X_np = np.random.randn(10, 5)
        Y_np = np.random.randn(3, 5)
        omegas_np = np.random.randn(3, 5, 2)

        X_jax = jnp.array(X_np)
        Y_jax = jnp.array(Y_np)
        omegas_jax = jnp.array(omegas_np)

        # JAX version
        D_jax = lomega_distance_matrix(X_jax, Y_jax, omegas_jax)

        # NumPy version
        D_numpy = np_lomega_distance(X_np, Y_np, omegas_np)

        np.testing.assert_allclose(D_jax, D_numpy, rtol=1e-4, atol=1e-6)


class TestGaussianKernel:
    """Test Gaussian kernel computations."""

    def test_self_kernel_is_one(self):
        """Test that K(x, x) = 1."""
        X = jnp.array([[1, 2, 3], [4, 5, 6]])
        sigma = 1.0

        K = gaussian_kernel_matrix(X, X, sigma)

        np.testing.assert_allclose(jnp.diag(K), 1.0, rtol=1e-5)

    def test_range_zero_to_one(self):
        """Test that kernel values are in [0, 1]."""
        np.random.seed(42)
        X = jnp.array(np.random.randn(20, 10))
        Y = jnp.array(np.random.randn(15, 10))
        sigma = 1.0

        K = gaussian_kernel_matrix(X, Y, sigma)

        assert jnp.all(K >= 0.0)
        assert jnp.all(K <= 1.0)

    def test_decreases_with_distance(self):
        """Test that kernel value decreases as distance increases."""
        X = jnp.array([[0, 0]])
        Y = jnp.array([[1, 0], [2, 0], [3, 0]])  # Increasing distance
        sigma = 1.0

        K = gaussian_kernel_matrix(X, Y, sigma)

        # Should decrease monotonically
        assert K[0, 0] > K[0, 1]
        assert K[0, 1] > K[0, 2]

    def test_symmetry(self):
        """Test that K(X, Y) = K(Y, X).T"""
        np.random.seed(42)
        X = jnp.array(np.random.randn(5, 3))
        Y = jnp.array(np.random.randn(4, 3))
        sigma = 1.0

        K_xy = gaussian_kernel_matrix(X, Y, sigma)
        K_yx = gaussian_kernel_matrix(Y, X, sigma)

        np.testing.assert_allclose(K_xy, K_yx.T, rtol=1e-5)

    def test_sigma_effect(self):
        """Test effect of sigma on kernel values."""
        X = jnp.array([[0, 0]])
        Y = jnp.array([[1, 0]])  # Distance = 1

        K_small = gaussian_kernel_matrix(X, Y, sigma=0.1)
        K_large = gaussian_kernel_matrix(X, Y, sigma=10.0)

        # Smaller sigma -> sharper kernel -> smaller values for distant points
        assert K_small[0, 0] < K_large[0, 0]


class TestPolynomialKernel:
    """Test polynomial kernel computations."""

    def test_linear_kernel(self):
        """Test that degree=1, coef0=0 gives linear kernel."""
        X = jnp.array([[1, 2], [3, 4]])
        Y = jnp.array([[1, 0], [0, 1]])

        K = polynomial_kernel_matrix(X, Y, degree=1, coef0=0.0)
        K_expected = X @ Y.T  # Dot product

        np.testing.assert_allclose(K, K_expected, rtol=1e-5)

    def test_degree_effect(self):
        """Test that higher degree gives different values."""
        X = jnp.array([[1, 2]])
        Y = jnp.array([[1, 1]])

        K_deg1 = polynomial_kernel_matrix(X, Y, degree=1, coef0=1.0)
        K_deg2 = polynomial_kernel_matrix(X, Y, degree=2, coef0=1.0)
        K_deg3 = polynomial_kernel_matrix(X, Y, degree=3, coef0=1.0)

        # All should be different
        assert not np.isclose(K_deg1[0, 0], K_deg2[0, 0])
        assert not np.isclose(K_deg2[0, 0], K_deg3[0, 0])


class TestUtilityFunctions:
    """Test utility functions."""

    def test_estimate_sigma(self):
        """Test sigma estimation."""
        np.random.seed(42)
        X = jnp.array(np.random.randn(100, 10))

        sigma = estimate_sigma(X, percentile=50.0)

        assert sigma > 0
        assert isinstance(sigma, float)

    def test_safe_divide(self):
        """Test safe division."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 0.0, 1.0])  # Contains zero

        result = safe_divide(x, y, epsilon=1e-10)

        assert jnp.all(jnp.isfinite(result))  # No inf
        assert not jnp.any(jnp.isnan(result))  # No nan

    def test_safe_divide_epsilon(self):
        """Test that epsilon affects result."""
        x = jnp.array([1.0])
        y = jnp.array([0.0])

        result1 = safe_divide(x, y, epsilon=1e-10)
        result2 = safe_divide(x, y, epsilon=1e-5)

        # Different epsilons should give different results
        assert not np.isclose(result1[0], result2[0])


class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_very_small_values(self):
        """Test with very small coordinate values."""
        X = jnp.array([[1e-10, 1e-10], [2e-10, 2e-10]])
        Y = jnp.array([[0, 0], [1e-10, 1e-10]])

        D = euclidean_distance_matrix(X, Y)

        assert jnp.all(jnp.isfinite(D))
        assert not jnp.any(jnp.isnan(D))

    def test_very_large_values(self):
        """Test with very large coordinate values."""
        X = jnp.array([[1e6, 1e6], [2e6, 2e6]])
        Y = jnp.array([[0, 0], [1e6, 1e6]])

        D = euclidean_distance_matrix(X, Y)

        assert jnp.all(jnp.isfinite(D))
        assert not jnp.any(jnp.isnan(D))

    def test_near_zero_distances(self):
        """Test with nearly identical points."""
        X = jnp.array([[1.0, 2.0], [1.0 + 1e-8, 2.0 + 1e-8]])

        D = euclidean_distance_matrix(X, X)

        assert jnp.all(jnp.isfinite(D))
        assert not jnp.any(jnp.isnan(D))
        assert jnp.all(D >= 0)  # No negative distances due to numerical error


class TestPerformance:
    """Basic performance tests."""

    @pytest.mark.slow
    def test_large_dataset(self):
        """Test on large dataset to ensure it completes."""
        np.random.seed(42)
        X = jnp.array(np.random.randn(10000, 50))
        Y = jnp.array(np.random.randn(100, 50))

        # Should complete without error
        D = euclidean_distance_matrix(X, Y)

        assert D.shape == (10000, 100)

    @pytest.mark.slow
    def test_jit_compilation(self):
        """Test that JIT compilation works."""
        X = jnp.array(np.random.randn(100, 10))
        Y = jnp.array(np.random.randn(50, 10))

        # First call: compilation + execution
        _ = euclidean_distance_matrix(X, Y)

        # Second call: should be faster (cached)
        import time
        start = time.time()
        _ = euclidean_distance_matrix(X, Y)
        jit_time = time.time() - start

        # Should be very fast (< 0.1s for this size)
        assert jit_time < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
