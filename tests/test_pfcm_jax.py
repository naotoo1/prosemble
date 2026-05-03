"""
Comprehensive tests for JAX-based Possibilistic Fuzzy C-Means (PFCM).

Tests cover:
1. Basic functionality (fit, predict, predict_proba)
2. Parameter validation
3. Numerical correctness
4. Edge cases
5. Reproducibility
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax

from prosemble.models import PFCM


class TestPFCMJAXBasic:
    """Test basic PFCM functionality."""

    def test_initialization(self):
        """Test PFCM initialization with valid parameters."""
        model = PFCM(n_clusters=3, fuzzifier=2.0, eta=2.0, a=1.0, b=1.0, k=1.0)
        assert model.n_clusters == 3
        assert model.fuzzifier == 2.0
        assert model.eta == 2.0
        assert model.a == 1.0
        assert model.b == 1.0
        assert model.k == 1.0
        assert model.max_iter == 100
        assert model.epsilon == 1e-5
        assert model.init_method == 'fcm'

    def test_initialization_custom_params(self):
        """Test PFCM initialization with custom parameters."""
        model = PFCM(
            n_clusters=5,
            fuzzifier=1.5,
            eta=2.5,
            a=2.0,
            b=2.0,
            k=0.5,
            max_iter=50,
            epsilon=1e-4,
            init_method='random',
            random_seed=123
        )
        assert model.n_clusters == 5
        assert model.fuzzifier == 1.5
        assert model.eta == 2.5
        assert model.a == 2.0
        assert model.b == 2.0
        assert model.k == 0.5
        assert model.max_iter == 50
        assert model.epsilon == 1e-4
        assert model.init_method == 'random'

    def test_invalid_n_clusters(self):
        """Test that invalid n_clusters raises ValueError."""
        with pytest.raises(ValueError, match="n_clusters must be >= 2"):
            PFCM(n_clusters=1)

    def test_invalid_fuzzifier(self):
        """Test that invalid fuzzifier raises ValueError."""
        with pytest.raises(ValueError, match="fuzzifier must be > 1"):
            PFCM(n_clusters=3, fuzzifier=1.0)

    def test_invalid_eta(self):
        """Test that invalid eta raises ValueError."""
        with pytest.raises(ValueError, match="eta must be > 1"):
            PFCM(n_clusters=3, eta=1.0)

    def test_invalid_k(self):
        """Test that invalid k raises ValueError."""
        with pytest.raises(ValueError, match="k must be > 0"):
            PFCM(n_clusters=3, k=0)

    def test_invalid_max_iter(self):
        """Test that invalid max_iter raises ValueError."""
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            PFCM(n_clusters=3, max_iter=0)

    def test_invalid_epsilon(self):
        """Test that invalid epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be > 0"):
            PFCM(n_clusters=3, epsilon=0)

    def test_invalid_init_method(self):
        """Test that invalid init_method raises ValueError."""
        with pytest.raises(ValueError, match="init_method must be"):
            PFCM(n_clusters=3, init_method='invalid')


class TestPFCMJAXFit:
    """Test PFCM fit method."""

    def test_fit_simple_data(self):
        """Test fitting on simple 2D data."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0]
        ])

        model = PFCM(n_clusters=2, fuzzifier=2.0, eta=2.0, a=1.0, b=1.0, k=1.0, random_seed=42)
        model.fit(X)

        assert model.centroids_ is not None
        assert model.U_ is not None
        assert model.T_ is not None
        assert model.gamma_ is not None
        assert model.n_iter_ is not None
        assert model.objective_ is not None

        assert model.centroids_.shape == (2, 2)
        assert model.U_.shape == (6, 2)
        assert model.T_.shape == (6, 2)
        assert model.gamma_.shape == (2,)

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        model = PFCM(n_clusters=2, random_seed=42)
        result = model.fit(X)
        assert result is model

    def test_fit_insufficient_samples(self):
        """Test that fitting with too few samples raises error."""
        X = jnp.array([[1, 2]])
        model = PFCM(n_clusters=2)
        with pytest.raises((ValueError, RuntimeError)):
            model.fit(X)

    def test_fit_updates_attributes(self):
        """Test that fit properly updates all model attributes."""
        X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        model = PFCM(n_clusters=2, random_seed=42)

        assert model.centroids_ is None
        assert model.U_ is None
        assert model.T_ is None
        assert model.gamma_ is None

        model.fit(X)
        assert model.centroids_ is not None
        assert model.U_ is not None
        assert model.T_ is not None
        assert model.gamma_ is not None
        assert isinstance(model.n_iter_, int)
        assert model.n_iter_ > 0

    def test_fit_fcm_initialization(self):
        """Test fitting with FCM initialization."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, init_method='fcm', random_seed=42)
        model.fit(X)

        assert model.centroids_ is not None
        assert model.U_ is not None
        assert model.T_ is not None

    def test_fit_random_initialization(self):
        """Test fitting with random initialization."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, init_method='random', random_seed=42)
        model.fit(X)

        assert model.centroids_ is not None
        assert model.U_ is not None
        assert model.T_ is not None


class TestPFCMJAXPredict:
    """Test PFCM predict method."""

    def test_predict_before_fit(self):
        """Test that predict before fit raises error."""
        X = jnp.array([[1, 2], [3, 4]])
        model = PFCM(n_clusters=2)
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_predict_returns_labels(self):
        """Test that predict returns integer labels."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)
        labels = model.predict(X)

        assert labels.shape == (4,)
        assert jnp.all((labels >= 0) & (labels < 2))

    def test_predict_consistency(self):
        """Test that predict is consistent (same input -> same output)."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)

        labels1 = model.predict(X)
        labels2 = model.predict(X)

        assert jnp.allclose(labels1, labels2)

    def test_predict_cluster_separation(self):
        """Test that predict separates well-separated clusters."""
        cluster1 = jnp.array([[0, 0], [0.1, 0.1], [0.2, 0.2]])
        cluster2 = jnp.array([[10, 10], [10.1, 10.1], [10.2, 10.2]])
        X = jnp.concatenate([cluster1, cluster2], axis=0)

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)
        labels = model.predict(X)

        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]


class TestPFCMJAXPredictProba:
    """Test PFCM predict_proba method."""

    def test_predict_proba_before_fit(self):
        """Test that predict_proba before fit raises error."""
        X = jnp.array([[1, 2], [3, 4]])
        model = PFCM(n_clusters=2)
        with pytest.raises(RuntimeError):
            model.predict_proba(X)

    def test_predict_proba_shape(self):
        """Test that predict_proba returns correct shape."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)
        proba = model.predict_proba(X)

        assert proba.shape == (4, 2)

    def test_predict_proba_valid_range(self):
        """Test that probability values are in [0, 1]."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)
        proba = model.predict_proba(X)

        assert jnp.all(proba >= 0)
        assert jnp.all(proba <= 1)

    def test_predict_proba_sum_to_one(self):
        """Test that fuzzy membership probabilities sum to 1 (U component)."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)

        U = model.U_
        row_sums = jnp.sum(U, axis=1)
        assert jnp.allclose(row_sums, 1.0)


class TestPFCMJAXTypicality:
    """Test PFCM typicality prediction."""

    def test_predict_typicality_before_fit(self):
        """Test that predict_typicality before fit raises error."""
        X = jnp.array([[1, 2], [3, 4]])
        model = PFCM(n_clusters=2)
        with pytest.raises(RuntimeError):
            model.predict_typicality(X)

    def test_predict_typicality_shape(self):
        """Test that predict_typicality returns correct shape."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)
        T = model.predict_typicality(X)

        assert T.shape == (4, 2)

    def test_predict_typicality_valid_range(self):
        """Test that typicality values are in [0, 1]."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)
        T = model.predict_typicality(X)

        assert jnp.all(T >= 0)
        assert jnp.all(T <= 1)

    def test_typicality_not_sum_to_one(self):
        """Test that typicality values don't necessarily sum to 1 (T component)."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)
        T = model.predict_typicality(X)

        row_sums = jnp.sum(T, axis=1)
        assert not jnp.allclose(row_sums, 1.0)


class TestPFCMJAXObjective:
    """Test PFCM objective function."""

    def test_objective_decreases(self):
        """Test that objective function generally decreases during training."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0]
        ])

        model = PFCM(n_clusters=2, max_iter=20, random_seed=42)
        model.fit(X)

        objectives = model.get_objective_history()

        # Objective should generally decrease, allow small tolerance for PFCM
        assert objectives[-1] <= objectives[0] * 1.05

        decreases = jnp.sum(jnp.diff(objectives) <= 0)
        total = len(objectives) - 1
        assert decreases / total >= 0.5

    def test_get_objective_history_before_fit(self):
        """Test that get_objective_history before fit raises error."""
        model = PFCM(n_clusters=2)
        with pytest.raises(RuntimeError):
            model.get_objective_history()

    def test_objective_history_length(self):
        """Test that objective history has correct length."""
        X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        max_iter = 10

        model = PFCM(n_clusters=2, max_iter=max_iter, random_seed=42)
        model.fit(X)

        objectives = model.get_objective_history()
        assert len(objectives) == max_iter


class TestPFCMJAXReproducibility:
    """Test PFCM reproducibility."""

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model1 = PFCM(n_clusters=2, random_seed=42)
        model1.fit(X)
        labels1 = model1.predict(X)
        centroids1 = model1.centroids_

        model2 = PFCM(n_clusters=2, random_seed=42)
        model2.fit(X)
        labels2 = model2.predict(X)
        centroids2 = model2.centroids_

        assert jnp.allclose(labels1, labels2)
        assert jnp.allclose(centroids1, centroids2, rtol=1e-5)

    def test_different_seeds_different_results(self):
        """Test that different seeds can produce different results."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model1 = PFCM(n_clusters=2, random_seed=42, init_method='random')
        model1.fit(X)
        centroids1 = model1.centroids_

        model2 = PFCM(n_clusters=2, random_seed=123, init_method='random')
        model2.fit(X)
        centroids2 = model2.centroids_

        assert not jnp.allclose(centroids1, centroids2, rtol=1e-2)


class TestPFCMJAXNumericalStability:
    """Test PFCM numerical stability."""

    def test_identical_points(self):
        """Test handling of identical points."""
        X = jnp.array([
            [1.0, 2.0],
            [1.0, 2.0],
            [5.0, 6.0],
            [5.0, 6.0]
        ])

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)
        labels = model.predict(X)

        assert labels[0] == labels[1]
        assert labels[2] == labels[3]

    def test_very_close_centroids(self):
        """Test handling when clusters are very close."""
        X = jnp.array([
            [1.0, 1.0],
            [1.01, 1.01],
            [1.02, 1.02],
            [1.03, 1.03]
        ])

        model = PFCM(n_clusters=2, random_seed=42)
        model.fit(X)
        labels = model.predict(X)

        assert labels.shape == (4,)

    def test_high_fuzzifier(self):
        """Test PFCM with high fuzzifier value."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        model = PFCM(n_clusters=2, fuzzifier=5.0, random_seed=42)
        model.fit(X)
        U = model.predict_proba(X)

        assert U.shape == (4, 2)
        assert jnp.all(U >= 0) and jnp.all(U <= 1)

    def test_different_weight_parameters(self):
        """Test PFCM with different a and b weight parameters produces different objectives."""
        from sklearn.datasets import load_iris
        X, _ = load_iris(return_X_y=True)
        X_jax = jnp.array(X)

        model1 = PFCM(n_clusters=3, a=10.0, b=0.1, random_seed=42)
        model1.fit(X_jax)

        model2 = PFCM(n_clusters=3, a=0.1, b=10.0, random_seed=42)
        model2.fit(X_jax)

        # Different weights should produce different final objectives
        assert not jnp.isclose(model1.objective_, model2.objective_, rtol=0.01)


class TestPFCMJAXRealData:
    """Test PFCM on real datasets."""

    def test_iris_dataset(self):
        """Test PFCM on Iris dataset."""
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        X_jax = jnp.array(X)

        model = PFCM(n_clusters=3, fuzzifier=2.0, eta=2.0, a=1.0, b=1.0, k=1.0, random_seed=42)
        model.fit(X_jax)

        assert model.centroids_.shape == (3, 4)
        assert model.U_.shape == (150, 3)
        assert model.T_.shape == (150, 3)
        assert model.gamma_.shape == (3,)

        labels = model.predict(X_jax)
        assert labels.shape == (150,)
        assert len(jnp.unique(labels)) <= 3

        U = model.predict_proba(X_jax)
        assert U.shape == (150, 3)
        assert jnp.all(U >= 0) and jnp.all(U <= 1)

        T = model.predict_typicality(X_jax)
        assert T.shape == (150, 3)
        assert jnp.all(T >= 0) and jnp.all(T <= 1)


class TestPFCMJAXDeviceCompatibility:
    """Test PFCM device compatibility."""

    def test_cpu_device(self):
        """Test that PFCM works on CPU."""
        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        with jax.default_device(jax.devices('cpu')[0]):
            model = PFCM(n_clusters=2, random_seed=42)
            model.fit(X)
            labels = model.predict(X)

        assert labels.shape == (4,)

    def test_gpu_device_if_available(self):
        """Test that PFCM works on GPU if available."""
        try:
            gpu_devices = jax.devices('gpu')
            if len(gpu_devices) == 0:
                pytest.skip("No GPU available")
        except RuntimeError:
            pytest.skip("No GPU backend available")

        X = jnp.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0]
        ])

        with jax.default_device(jax.devices('gpu')[0]):
            X_gpu = jax.device_put(X, jax.devices('gpu')[0])
            model = PFCM(n_clusters=2, random_seed=42)
            model.fit(X_gpu)
            labels = model.predict(X_gpu)

        assert labels.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
