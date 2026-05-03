"""
JAX-based distance functions test suite.

Tests cover:
- Matrix distance computations (vectorized)
- Pairwise distance computations
- Kernel functions (Gaussian, Polynomial)
- Numerical stability
- GPU/CPU compatibility
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from prosemble.core import (
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

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = jnp.array(np.random.randn(32, 6), dtype=jnp.float32)
    Y = jnp.array(np.random.randn(8, 6), dtype=jnp.float32)
    return X, Y


@pytest.fixture
def small_data():
    """Small data for quick tests."""
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Y = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    return X, Y


class TestEuclideanDistanceMatrix:
    """Test Euclidean distance matrix computation."""

    def test_output_shape(self, sample_data):
        """Test correct output shape."""
        X, Y = sample_data
        D = euclidean_distance_matrix(X, Y)
        assert D.shape == (32, 8)

    def test_self_distance_zero(self, small_data):
        """Test distance from point to itself is zero."""
        X, _ = small_data
        D = euclidean_distance_matrix(X, X)
        np.testing.assert_allclose(jnp.diag(D), 0.0, atol=1e-6)

    def test_known_values(self):
        """Test against known distance values."""
        X = jnp.array([[0.0, 0.0]])
        Y = jnp.array([[3.0, 4.0]])  # 3-4-5 triangle
        D = euclidean_distance_matrix(X, Y)
        assert jnp.isclose(D[0, 0], 5.0, atol=1e-6)

    def test_symmetry(self, small_data):
        """Test D(X, Y) = D(Y, X).T"""
        X, Y = small_data
        D_xy = euclidean_distance_matrix(X, Y)
        D_yx = euclidean_distance_matrix(Y, X)
        np.testing.assert_allclose(D_xy, D_yx.T, rtol=1e-5)

    def test_non_negative(self, sample_data):
        """Test all distances are non-negative."""
        X, Y = sample_data
        D = euclidean_distance_matrix(X, Y)
        assert jnp.all(D >= 0)


class TestSquaredEuclideanDistanceMatrix:
    """Test squared Euclidean distance matrix."""

    def test_squared_relationship(self, small_data):
        """Test D_sq = D²."""
        X, Y = small_data
        D = euclidean_distance_matrix(X, Y)
        D_sq = squared_euclidean_distance_matrix(X, Y)
        np.testing.assert_allclose(D ** 2, D_sq, rtol=1e-5)

    def test_non_negative(self, sample_data):
        """Test all squared distances are non-negative."""
        X, Y = sample_data
        D_sq = squared_euclidean_distance_matrix(X, Y)
        assert jnp.all(D_sq >= 0)


class TestManhattanDistanceMatrix:
    """Test Manhattan distance matrix."""

    def test_known_values(self):
        """Test against known Manhattan distance."""
        X = jnp.array([[0.0, 0.0]])
        Y = jnp.array([[3.0, 4.0]])
        D = manhattan_distance_matrix(X, Y)
        assert jnp.isclose(D[0, 0], 7.0)  # |3-0| + |4-0| = 7

    def test_unit_hypercube(self):
        """Test on unit hypercube corners."""
        X = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        D = manhattan_distance_matrix(X, X)

        # Diagonal should be zero
        np.testing.assert_allclose(jnp.diag(D), 0.0, atol=1e-6)

        # Adjacent corners should have distance 1
        assert jnp.isclose(D[0, 1], 1.0)
        assert jnp.isclose(D[0, 2], 1.0)

        # Opposite corners should have distance 2
        assert jnp.isclose(D[0, 3], 2.0)


class TestLpNormDistanceMatrix:
    """Test Lp-norm distance matrix."""

    @pytest.mark.parametrize("p", [1, 2, 3, jnp.inf])
    def test_different_p_values(self, small_data, p):
        """Test Lp-norm with different p values."""
        X, Y = small_data
        D = lpnorm_distance_matrix(X, Y, p=p)
        assert D.shape == (3, 2)
        assert jnp.all(D >= 0)

    def test_l1_equals_manhattan(self, small_data):
        """Test L1 norm equals Manhattan distance."""
        X, Y = small_data
        D_l1 = lpnorm_distance_matrix(X, Y, p=1)
        D_manhattan = manhattan_distance_matrix(X, Y)
        np.testing.assert_allclose(D_l1, D_manhattan, rtol=1e-5)

    def test_l2_equals_euclidean(self, small_data):
        """Test L2 norm equals Euclidean distance."""
        X, Y = small_data
        D_l2 = lpnorm_distance_matrix(X, Y, p=2)
        D_euclidean = euclidean_distance_matrix(X, Y)
        np.testing.assert_allclose(D_l2, D_euclidean, rtol=1e-5)

    def test_linf_is_max(self):
        """Test L-infinity norm is max absolute difference."""
        X = jnp.array([[0.0, 0.0, 0.0]])
        Y = jnp.array([[1.0, 5.0, 3.0]])  # max diff is 5
        D = lpnorm_distance_matrix(X, Y, p=jnp.inf)
        assert jnp.isclose(D[0, 0], 5.0)


class TestOmegaDistanceMatrix:
    """Test Omega (projection-based) distance matrix."""

    def test_identity_omega_squared_euclidean(self, small_data):
        """Test identity projection gives squared Euclidean."""
        X, Y = small_data
        d = X.shape[1]
        omega = jnp.eye(d)

        D_omega = omega_distance_matrix(X, Y, omega)
        D_sq = squared_euclidean_distance_matrix(X, Y)

        np.testing.assert_allclose(D_omega, D_sq, rtol=1e-5)

    def test_projection_reduces_dimension(self):
        """Test projection to lower dimension."""
        X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Y = jnp.array([[0.0, 0.0, 0.0]])

        # Project to first 2 dimensions only
        omega = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        D = omega_distance_matrix(X, Y, omega)

        # Manually compute expected result
        X_proj = X @ omega  # [[1, 2], [4, 5]]
        Y_proj = Y @ omega  # [[0, 0]]
        expected = squared_euclidean_distance_matrix(X_proj, Y_proj)

        np.testing.assert_allclose(D, expected, rtol=1e-5)


class TestLomegaDistanceMatrix:
    """Test local omega (multiple projections) distance matrix."""

    def test_output_shape(self, small_data):
        """Test correct output shape."""
        X, Y = small_data
        n_y = Y.shape[0]
        d = X.shape[1]

        # Create random projection matrices
        omegas = jax.random.normal(jax.random.PRNGKey(42), (n_y, d, 2))

        D = lomega_distance_matrix(X, Y, omegas)
        assert D.shape == (X.shape[0], Y.shape[0])

    def test_identity_omegas_squared_euclidean(self, small_data):
        """Test identity projections give squared Euclidean."""
        X, Y = small_data
        d = X.shape[1]
        n_y = Y.shape[0]

        # Stack identity matrices
        omega = jnp.eye(d)
        omegas = jnp.stack([omega for _ in range(n_y)], axis=0)

        D_lomega = lomega_distance_matrix(X, Y, omegas)
        D_sq = squared_euclidean_distance_matrix(X, Y)

        np.testing.assert_allclose(D_lomega, D_sq, rtol=1e-5)


class TestGaussianKernelMatrix:
    """Test Gaussian kernel matrix."""

    def test_self_kernel_is_one(self, small_data):
        """Test K(x, x) = 1."""
        X, _ = small_data
        sigma = 1.0
        K = gaussian_kernel_matrix(X, X, sigma)
        np.testing.assert_allclose(jnp.diag(K), 1.0, rtol=1e-5)

    def test_range_zero_to_one(self, sample_data):
        """Test kernel values in [0, 1]."""
        X, Y = sample_data
        sigma = 1.0
        K = gaussian_kernel_matrix(X, Y, sigma)

        assert jnp.all(K >= 0.0)
        assert jnp.all(K <= 1.0)

    def test_decreases_with_distance(self):
        """Test kernel decreases as distance increases."""
        X = jnp.array([[0.0, 0.0]])
        Y = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        sigma = 1.0

        K = gaussian_kernel_matrix(X, Y, sigma)

        # Should decrease monotonically
        assert K[0, 0] > K[0, 1]
        assert K[0, 1] > K[0, 2]

    def test_symmetry(self, small_data):
        """Test K(X, Y) = K(Y, X).T"""
        X, Y = small_data
        sigma = 1.0

        K_xy = gaussian_kernel_matrix(X, Y, sigma)
        K_yx = gaussian_kernel_matrix(Y, X, sigma)

        np.testing.assert_allclose(K_xy, K_yx.T, rtol=1e-5)

    def test_sigma_effect(self):
        """Test effect of sigma on kernel values."""
        X = jnp.array([[0.0, 0.0]])
        Y = jnp.array([[1.0, 0.0]])  # Distance = 1

        K_small = gaussian_kernel_matrix(X, Y, sigma=0.1)
        K_large = gaussian_kernel_matrix(X, Y, sigma=10.0)

        # Smaller sigma -> sharper kernel -> smaller values for distant points
        assert K_small[0, 0] < K_large[0, 0]


class TestPolynomialKernelMatrix:
    """Test polynomial kernel matrix."""

    def test_linear_kernel(self):
        """Test degree=1, coef0=0 gives linear kernel."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        Y = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        K = polynomial_kernel_matrix(X, Y, degree=1, coef0=0.0)
        K_expected = X @ Y.T  # Dot product

        np.testing.assert_allclose(K, K_expected, rtol=1e-5)

    def test_degree_effect(self):
        """Test higher degree gives different values."""
        X = jnp.array([[1.0, 2.0]])
        Y = jnp.array([[1.0, 1.0]])

        K_deg1 = polynomial_kernel_matrix(X, Y, degree=1, coef0=1.0)
        K_deg2 = polynomial_kernel_matrix(X, Y, degree=2, coef0=1.0)
        K_deg3 = polynomial_kernel_matrix(X, Y, degree=3, coef0=1.0)

        # All should be different
        assert not jnp.isclose(K_deg1[0, 0], K_deg2[0, 0])
        assert not jnp.isclose(K_deg2[0, 0], K_deg3[0, 0])


class TestPairwiseDistanceFunctions:
    """Test pairwise (non-matrix) distance functions."""

    def test_euclidean_distance(self):
        """Test pairwise Euclidean distance."""
        x = jnp.array([0.0, 0.0])
        y = jnp.array([3.0, 4.0])

        d = euclidean_distance(x, y)
        assert jnp.isclose(d, 5.0)

    def test_squared_euclidean_distance(self):
        """Test pairwise squared Euclidean distance."""
        x = jnp.array([0.0, 0.0])
        y = jnp.array([3.0, 4.0])

        d_sq = squared_euclidean_distance(x, y)
        assert jnp.isclose(d_sq, 25.0)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_estimate_sigma(self, sample_data):
        """Test sigma estimation."""
        X, _ = sample_data
        sigma = estimate_sigma(X, percentile=50.0)

        assert sigma > 0
        assert isinstance(float(sigma), float)

    def test_estimate_sigma_different_percentiles(self, sample_data):
        """Test different percentiles give different sigmas."""
        X, _ = sample_data

        sigma_25 = estimate_sigma(X, percentile=25.0)
        sigma_50 = estimate_sigma(X, percentile=50.0)
        sigma_75 = estimate_sigma(X, percentile=75.0)

        # Higher percentile -> larger sigma
        assert sigma_25 < sigma_50 < sigma_75

    def test_safe_divide(self):
        """Test safe division."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 0.0, 1.0])  # Contains zero

        result = safe_divide(x, y, epsilon=1e-10)

        assert jnp.all(jnp.isfinite(result))
        assert not jnp.any(jnp.isnan(result))

    def test_safe_divide_epsilon_effect(self):
        """Test epsilon affects result."""
        x = jnp.array([1.0])
        y = jnp.array([0.0])

        result1 = safe_divide(x, y, epsilon=1e-10)
        result2 = safe_divide(x, y, epsilon=1e-5)

        # Different epsilons should give different results
        assert not jnp.isclose(result1[0], result2[0])


class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_very_small_values(self):
        """Test with very small coordinate values."""
        X = jnp.array([[1e-10, 1e-10], [2e-10, 2e-10]])
        Y = jnp.array([[0.0, 0.0], [1e-10, 1e-10]])

        D = euclidean_distance_matrix(X, Y)

        assert jnp.all(jnp.isfinite(D))
        assert not jnp.any(jnp.isnan(D))

    def test_very_large_values(self):
        """Test with very large coordinate values."""
        X = jnp.array([[1e6, 1e6], [2e6, 2e6]])
        Y = jnp.array([[0.0, 0.0], [1e6, 1e6]])

        D = euclidean_distance_matrix(X, Y)

        assert jnp.all(jnp.isfinite(D))
        assert not jnp.any(jnp.isnan(D))

    def test_near_zero_distances(self):
        """Test with nearly identical points."""
        X = jnp.array([[1.0, 2.0], [1.0 + 1e-8, 2.0 + 1e-8]])
        D = euclidean_distance_matrix(X, X)

        assert jnp.all(jnp.isfinite(D))
        assert not jnp.any(jnp.isnan(D))
        assert jnp.all(D >= 0)  # No negative distances

    def test_kernel_with_small_sigma(self):
        """Test Gaussian kernel with very small sigma."""
        X = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        Y = jnp.array([[0.0, 0.0]])

        K = gaussian_kernel_matrix(X, Y, sigma=1e-5)

        assert jnp.all(jnp.isfinite(K))
        assert jnp.all(K >= 0) and jnp.all(K <= 1)


class TestDeviceCompatibility:
    """Test CPU/GPU device compatibility."""

    def test_cpu_device(self, small_data):
        """Test functions work on CPU."""
        X, Y = small_data

        with jax.default_device(jax.devices('cpu')[0]):
            D = euclidean_distance_matrix(X, Y)
            assert D.shape == (3, 2)

    def test_gpu_device_if_available(self, small_data):
        """Test functions work on GPU if available."""
        try:
            gpu_devices = jax.devices('gpu')
            if len(gpu_devices) == 0:
                pytest.skip("No GPU available")
        except RuntimeError:
            pytest.skip("No GPU backend available")

        X, Y = small_data

        with jax.default_device(gpu_devices[0]):
            X_gpu = jax.device_put(X, gpu_devices[0])
            Y_gpu = jax.device_put(Y, gpu_devices[0])

            D = euclidean_distance_matrix(X_gpu, Y_gpu)
            assert D.shape == (3, 2)


class TestJITCompilation:
    """Test JIT compilation works correctly."""

    def test_jit_compilation_euclidean(self, sample_data):
        """Test JIT compilation for Euclidean distance."""
        X, Y = sample_data

        # First call: compilation + execution
        D1 = euclidean_distance_matrix(X, Y)

        # Second call: should use cached compiled version
        D2 = euclidean_distance_matrix(X, Y)

        np.testing.assert_allclose(D1, D2, rtol=1e-5)

    def test_jit_compilation_kernel(self, sample_data):
        """Test JIT compilation for Gaussian kernel."""
        X, Y = sample_data

        # First call
        K1 = gaussian_kernel_matrix(X, Y, sigma=1.0)

        # Second call (cached)
        K2 = gaussian_kernel_matrix(X, Y, sigma=1.0)

        np.testing.assert_allclose(K1, K2, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
