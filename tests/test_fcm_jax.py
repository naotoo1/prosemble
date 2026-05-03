"""
Tests for JAX implementation of Fuzzy C-Means (FCM).

This test suite validates:
1. Correctness: JAX vs NumPy FCM parity
2. API compatibility
3. Numerical stability
4. Performance
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from prosemble.models import FCM
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    FCM = None

# Skip all tests if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


def sort_centroids(centroids, labels=None):
    """
    Sort centroids by first feature for comparison.

    FCM can produce equivalent solutions with permuted cluster labels.
    We sort to enable comparison.
    """
    indices = np.argsort(centroids[:, 0])
    return centroids[indices]


class TestFCMBasicFunctionality:
    """Test basic FCM functionality."""

    def test_fit_simple_data(self):
        """Test fitting on simple 2D data."""
        # Create simple clustered data
        X = jnp.array([
            [1, 2], [1.5, 1.8], [1, 0.6],  # Cluster 1
            [5, 8], [8, 8], [9, 11]         # Cluster 2
        ])

        model = FCM(n_clusters=2, fuzzifier=2.0, max_iter=100, random_seed=42)
        model.fit(X)

        assert model.centroids_ is not None
        assert model.U_ is not None
        assert model.centroids_.shape == (2, 2)
        assert model.U_.shape == (6, 2)

    def test_predict(self):
        """Test prediction on fitted model."""
        X = jnp.array([
            [1, 2], [1.5, 1.8],
            [5, 8], [8, 8]
        ])

        model = FCM(n_clusters=2, random_seed=42)
        model.fit(X)

        labels = model.predict(X)

        assert labels.shape == (4,)
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 2)

    def test_predict_proba(self):
        """Test fuzzy membership prediction."""
        X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])

        model = FCM(n_clusters=2, random_seed=42)
        model.fit(X)

        U = model.predict_proba(X)

        assert U.shape == (4, 2)
        # Check membership properties
        assert jnp.all(U >= 0)
        assert jnp.all(U <= 1)
        # Rows should sum to 1
        np.testing.assert_allclose(jnp.sum(U, axis=1), 1.0, rtol=1e-5)

    def test_final_centroids(self):
        """Test getting final centroids."""
        X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])

        model = FCM(n_clusters=2, random_seed=42)
        model.fit(X)

        centroids = model.final_centroids()

        assert centroids.shape == (2, 2)
        assert jnp.all(jnp.isfinite(centroids))

    def test_get_distance_space(self):
        """Test distance computation."""
        X = jnp.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])

        model = FCM(n_clusters=2, random_seed=42)
        model.fit(X)

        D = model.get_distance_space(X)

        assert D.shape == (4, 2)
        assert jnp.all(D >= 0)


class TestFCMObjectiveFunction:
    """Test objective function computation."""

    def test_objective_decreases(self):
        """Test that objective function decreases during training."""
        X = jnp.array(np.random.randn(100, 10).astype(np.float32))

        model = FCM(n_clusters=3, max_iter=50, random_seed=42)
        model.fit(X)

        objectives = model.get_objective_history()

        # Objective should generally decrease
        # (May not be strictly monotonic due to numerical precision)
        assert objectives[0] > objectives[-1]

    def test_objective_finite(self):
        """Test that objective function is always finite."""
        X = jnp.array(np.random.randn(50, 5).astype(np.float32))

        model = FCM(n_clusters=3, random_seed=42)
        model.fit(X)

        objectives = model.get_objective_history()

        assert jnp.all(jnp.isfinite(objectives))


class TestFCMParameterValidation:
    """Test parameter validation."""

    def test_invalid_n_clusters(self):
        """Test error for invalid n_clusters."""
        with pytest.raises(ValueError, match="n_clusters must be >= 2"):
            FCM(n_clusters=1)

    def test_invalid_fuzzifier(self):
        """Test error for invalid fuzzifier."""
        with pytest.raises(ValueError, match="fuzzifier must be > 1"):
            FCM(n_clusters=2, fuzzifier=1.0)

    def test_invalid_max_iter(self):
        """Test error for invalid max_iter."""
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            FCM(n_clusters=2, max_iter=0)

    def test_invalid_epsilon(self):
        """Test error for invalid epsilon."""
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            FCM(n_clusters=2, epsilon=0)

    def test_fit_wrong_shape(self):
        """Test error for wrong data shape."""
        X = jnp.array([1, 2, 3, 4])  # 1D array

        model = FCM(n_clusters=2)
        with pytest.raises(ValueError, match="X must be 2D"):
            model.fit(X)

    def test_fit_too_few_samples(self):
        """Test error when n_samples < n_clusters."""
        X = jnp.array([[1, 2], [3, 4]])  # Only 2 samples

        model = FCM(n_clusters=3)
        with pytest.raises(ValueError, match="n_samples .* must be >= n_clusters"):
            model.fit(X)

    def test_fit_nan_data(self):
        """Test error for NaN in data."""
        X = jnp.array([[1, 2], [3, jnp.nan], [5, 6]])

        model = FCM(n_clusters=2)
        with pytest.raises(ValueError, match="NaN or Inf"):
            model.fit(X)

    def test_predict_before_fit(self):
        """Test error when predicting before fitting."""
        X = jnp.array([[1, 2], [3, 4]])

        model = FCM(n_clusters=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)


class TestFCMNumPyParity:
    """Test parity between JAX and NumPy implementations."""

    # NOTE: This test has been commented out because the old NumPy FCM model
    # is no longer available after the JAX migration. The FCM class now refers
    # to the JAX implementation.
    #
    # def test_simple_clustering_parity(self):
    #     """Test that JAX and NumPy give similar results."""
    #     # Use small simple dataset
    #     np.random.seed(42)
    #     X_np = np.concatenate([
    #         np.random.randn(20, 2) + [0, 0],
    #         np.random.randn(20, 2) + [5, 5],
    #         np.random.randn(20, 2) + [0, 5]
    #     ]).astype(np.float32)
    #
    #     # NumPy version
    #     fcm_np = FCM(
    #         data=X_np,
    #         c=3,
    #         m=2.0,
    #         num_iter=100,
    #         epsilon=1e-5,
    #         ord='fro',
    #         set_U_matrix=None
    #     )
    #     fcm_np.fit()
    #     centroids_np = fcm_np.final_centroids()
    #     labels_np = fcm_np.predict()
    #
    #     # JAX version
    #     X_jax = jnp.array(X_np)
    #     fcm_jax = FCM(
    #         n_clusters=3,
    #         fuzzifier=2.0,
    #         max_iter=100,
    #         epsilon=1e-5,
    #         random_seed=42
    #     )
    #     fcm_jax.fit(X_jax)
    #     centroids_jax = fcm_jax.final_centroids()
    #     labels_jax = fcm_jax.predict(X_jax)
    #
    #     # Sort centroids for comparison (label switching doesn't matter)
    #     centroids_np_sorted = sort_centroids(centroids_np)
    #     centroids_jax_sorted = sort_centroids(np.array(centroids_jax))
    #
    #     # Centroids should be close
    #     np.testing.assert_allclose(
    #         centroids_np_sorted,
    #         centroids_jax_sorted,
    #         rtol=0.1,  # Allow 10% relative difference
    #         atol=0.5    # Allow 0.5 absolute difference
    #     )

    def test_membership_matrix_properties(self):
        """Test that membership matrix satisfies FCM properties."""
        X = jnp.array(np.random.randn(50, 5).astype(np.float32))

        model = FCM(n_clusters=3, random_seed=42)
        model.fit(X)

        U = model.U_

        # Property 1: All memberships in [0, 1]
        assert jnp.all(U >= 0)
        assert jnp.all(U <= 1)

        # Property 2: Each row sums to 1
        row_sums = jnp.sum(U, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

        # Property 3: Each column has at least some membership
        col_sums = jnp.sum(U, axis=0)
        assert jnp.all(col_sums > 0)


class TestFCMNumericalStability:
    """Test numerical stability in edge cases."""

    def test_identical_points(self):
        """Test with identical data points."""
        X = jnp.array([[1, 2]] * 10)  # All same point

        model = FCM(n_clusters=2, random_seed=42)
        model.fit(X)

        # Should not crash and produce finite results
        assert jnp.all(jnp.isfinite(model.centroids_))
        assert jnp.all(jnp.isfinite(model.U_))

    def test_very_close_points(self):
        """Test with very close but not identical points."""
        X = jnp.array([[1, 2], [1 + 1e-8, 2 + 1e-8], [1 + 2e-8, 2 + 2e-8]])

        model = FCM(n_clusters=2, random_seed=42)
        model.fit(X)

        assert jnp.all(jnp.isfinite(model.centroids_))
        assert jnp.all(jnp.isfinite(model.U_))

    def test_large_values(self):
        """Test with large coordinate values."""
        X = jnp.array(np.random.randn(50, 5).astype(np.float32)) * 1e6

        model = FCM(n_clusters=3, random_seed=42)
        model.fit(X)

        assert jnp.all(jnp.isfinite(model.centroids_))
        assert jnp.all(jnp.isfinite(model.U_))

    def test_small_values(self):
        """Test with very small coordinate values."""
        X = jnp.array(np.random.randn(50, 5).astype(np.float32)) * 1e-6

        model = FCM(n_clusters=3, random_seed=42)
        model.fit(X)

        assert jnp.all(jnp.isfinite(model.centroids_))
        assert jnp.all(jnp.isfinite(model.U_))


class TestFCMDifferentFuzzifiers:
    """Test FCM with different fuzzifier values."""

    def test_fuzzifier_effects(self):
        """Test that different fuzzifiers give different results."""
        X = jnp.array(np.random.randn(50, 5).astype(np.float32))

        # m = 1.5 (less fuzzy)
        model_15 = FCM(n_clusters=3, fuzzifier=1.5, random_seed=42)
        model_15.fit(X)
        U_15 = model_15.U_

        # m = 2.0 (standard)
        model_20 = FCM(n_clusters=3, fuzzifier=2.0, random_seed=42)
        model_20.fit(X)
        U_20 = model_20.U_

        # m = 3.0 (more fuzzy)
        model_30 = FCM(n_clusters=3, fuzzifier=3.0, random_seed=42)
        model_30.fit(X)
        U_30 = model_30.U_

        # Higher fuzzifier should give more uniform memberships
        # Check entropy as a measure of fuzziness
        def entropy(U):
            # Avoid log(0)
            U_safe = jnp.maximum(U, 1e-10)
            return -jnp.sum(U * jnp.log(U_safe))

        entropy_15 = entropy(U_15)
        entropy_20 = entropy(U_20)
        entropy_30 = entropy(U_30)

        # Higher fuzzifier -> higher entropy (more uniform membership)
        assert entropy_30 > entropy_20 > entropy_15


class TestFCMReproducibility:
    """Test reproducibility with random seeds."""

    def test_same_seed_same_result(self):
        """Test that same seed gives same result."""
        X = jnp.array(np.random.randn(50, 5).astype(np.float32))

        # First run
        model1 = FCM(n_clusters=3, random_seed=42)
        model1.fit(X)
        centroids1 = model1.centroids_

        # Second run with same seed
        model2 = FCM(n_clusters=3, random_seed=42)
        model2.fit(X)
        centroids2 = model2.centroids_

        np.testing.assert_allclose(centroids1, centroids2, rtol=1e-5)

    def test_different_seed_different_result(self):
        """Test that different seeds give different results."""
        X = jnp.array(np.random.randn(50, 5).astype(np.float32))

        # First run
        model1 = FCM(n_clusters=3, random_seed=42)
        model1.fit(X)
        centroids1 = model1.centroids_

        # Second run with different seed
        model2 = FCM(n_clusters=3, random_seed=123)
        model2.fit(X)
        centroids2 = model2.centroids_

        # Should be different (very unlikely to be same)
        assert not jnp.allclose(centroids1, centroids2, rtol=1e-3)


class TestFCMPredictNewData:
    """Test prediction on new unseen data."""

    def test_predict_new_samples(self):
        """Test predicting on new samples."""
        # Training data
        X_train = jnp.array([
            [1, 2], [1.5, 1.8], [1, 0.6],
            [5, 8], [8, 8], [9, 11]
        ])

        # New test data
        X_test = jnp.array([[1.2, 1.5], [7, 9]])

        model = FCM(n_clusters=2, random_seed=42)
        model.fit(X_train)

        # Predict on new data
        labels = model.predict(X_test)
        U = model.predict_proba(X_test)

        assert labels.shape == (2,)
        assert U.shape == (2, 2)
        np.testing.assert_allclose(jnp.sum(U, axis=1), 1.0, rtol=1e-5)


class TestFCMPerformance:
    """Basic performance tests."""

    @pytest.mark.slow
    def test_large_dataset(self):
        """Test on large dataset to ensure it completes."""
        X = jnp.array(np.random.randn(10000, 50).astype(np.float32))

        model = FCM(n_clusters=5, max_iter=50, random_seed=42)
        model.fit(X)

        assert model.centroids_.shape == (5, 50)
        assert model.U_.shape == (10000, 5)

    @pytest.mark.slow
    def test_high_dimensional(self):
        """Test on high-dimensional data."""
        X = jnp.array(np.random.randn(1000, 500).astype(np.float32))

        model = FCM(n_clusters=3, max_iter=50, random_seed=42)
        model.fit(X)

        assert model.centroids_.shape == (3, 500)


class TestFCMIrisDataset:
    """Test on classic Iris dataset."""

    def test_iris_clustering(self):
        """Test FCM on Iris dataset."""
        try:
            from sklearn.datasets import load_iris
        except ImportError:
            pytest.skip("scikit-learn not installed")

        X, y = load_iris(return_X_y=True)
        X = jnp.array(X.astype(np.float32))

        model = FCM(n_clusters=3, random_seed=42)
        model.fit(X)

        labels = model.predict(X)

        # Should produce 3 clusters
        unique_labels = jnp.unique(labels)
        assert len(unique_labels) == 3

        # Centroids should be reasonable
        assert model.centroids_.shape == (3, 4)
        assert jnp.all(jnp.isfinite(model.centroids_))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
