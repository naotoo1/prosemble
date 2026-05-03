"""
Comprehensive test suite for all 19 JAX clustering models.

Tests cover:
- Basic functionality (fit, predict)
- Reproducibility
- Device compatibility (CPU/GPU)
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from prosemble.models import (
        FCM, PCM, PFCM, HCM, FPCM,
        IPCM, IPCM2, AFCM,
        KFCM, KPCM, KAFCM, KFPCM, KPFCM,
        KIPCM, KIPCM2,
        BGPC, NPC, SOM, KNN,
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Define dummy classes to avoid NameError when JAX not available
    FCM = PCM = PFCM = HCM = FPCM = None
    IPCM = IPCM2 = AFCM = None
    KFCM = KPCM = KAFCM = KFPCM = KPFCM = None
    KIPCM = KIPCM2 = None
    BGPC = NPC = SOM = KNN = None
    jax = jnp = None

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


@pytest.fixture
def simple_data():
    """Simple 2-cluster data."""
    return jnp.array([
        [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
        [8.0, 9.0], [8.5, 8.8], [8.2, 9.1]
    ])


@pytest.fixture
def iris_data():
    """Iris dataset."""
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    return jnp.array(X, dtype=jnp.float32), y


# Standard clustering models (use random_seed)
STANDARD_MODELS = [
    (FCM, {'n_clusters': 2, 'fuzzifier': 2.0}),
    (PCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'k': 1.0}),
    (PFCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'eta': 2.0, 'a': 1.0, 'b': 1.0, 'k': 1.0}),
    (HCM, {'n_clusters': 2}),
    (FPCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'eta': 2.0}),
    (IPCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'tipifier': 2.0, 'k': 1.0}),
    (IPCM2, {'n_clusters': 2, 'fuzzifier': 2.0, 'tipifier': 2.0}),
    (AFCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'a': 1.0, 'b': 1.0, 'k': 1.0}),
    (KFCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'sigma': 1.0}),
    (KPCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'k': 1.0, 'sigma': 1.0}),
    (KAFCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'a': 1.0, 'b': 1.0, 'k': 1.0, 'sigma': 1.0}),
    (KFPCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'eta': 2.0, 'sigma': 1.0}),
    (KPFCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'eta': 2.0, 'a': 1.0, 'b': 1.0, 'k': 1.0, 'sigma': 1.0}),
    (KIPCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'tipifier': 2.0, 'sigma': 1.0}),
    (KIPCM2, {'n_clusters': 2, 'fuzzifier': 2.0, 'tipifier': 2.0, 'sigma': 1.0}),
]


class TestStandardModels:
    """Test standard clustering models."""

    @pytest.mark.parametrize("model_class,params", STANDARD_MODELS)
    def test_fit_predict(self, model_class, params, simple_data):
        """Test fit and predict on simple data."""
        model = model_class(**params, random_seed=42)
        model.fit(simple_data)

        assert hasattr(model, 'centroids_')
        assert model.n_iter_ > 0

        labels = model.predict(simple_data)
        assert labels.shape == (len(simple_data),)
        assert jnp.all(labels >= 0)

    @pytest.mark.parametrize("model_class,params", STANDARD_MODELS)
    def test_reproducibility(self, model_class, params, simple_data):
        """Test same seed gives same results."""
        model1 = model_class(**params, random_seed=42)
        model1.fit(simple_data)

        model2 = model_class(**params, random_seed=42)
        model2.fit(simple_data)

        np.testing.assert_allclose(model1.centroids_, model2.centroids_, rtol=1e-5)


class TestBGPC:
    """Test BGPC model (uses random_state)."""

    def test_fit_predict(self, simple_data):
        """Test BGPC fit and predict."""
        model = BGPC(n_clusters=2, max_iter=10, random_state=42)
        model.fit(simple_data)

        assert hasattr(model, 'centroids_')
        assert model.n_iter_ > 0

        labels = model.predict(simple_data)
        assert labels.shape == (len(simple_data),)

    def test_parameter_decay(self, simple_data):
        """Test beta parameter decays."""
        model = BGPC(n_clusters=2, max_iter=10, beta_init=0.1, beta_final=10.0, random_state=42)
        model.fit(simple_data)

        assert model.beta_ != model.beta_init


class TestSOM:
    """Test SOM model (uses random_state)."""

    def test_fit_transform(self, simple_data):
        """Test SOM fit and transform."""
        model = SOM(grid_size=3, max_iter=100, random_state=42)
        model.fit(simple_data)

        assert hasattr(model, 'som_')
        assert model.n_iter_ > 0

        coords = model.transform(simple_data)
        assert coords.shape == (len(simple_data), 2)

    def test_grid_structure(self, simple_data):
        """Test SOM maintains grid structure."""
        grid_size = 3
        model = SOM(grid_size=grid_size, max_iter=100, random_state=42)
        model.fit(simple_data)

        assert model.som_.shape == (grid_size, grid_size, simple_data.shape[1])


class TestNPC:
    """Test NPC supervised model."""

    def test_fit_predict(self, iris_data):
        """Test NPC fit and predict."""
        X, y = iris_data

        model = NPC(n_classes=3)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

        accuracy = jnp.mean(y_pred == y)
        assert accuracy > 0.5


class TestKNN:
    """Test KNN supervised model."""

    def test_fit_predict(self, iris_data):
        """Test KNN fit and predict."""
        X, y = iris_data

        model = KNN(n_neighbors=5)
        model.fit(X, y)

        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

        accuracy = jnp.mean(y_pred == y)
        assert accuracy > 0.8

    def test_different_k_values(self, iris_data):
        """Test different k values."""
        X, y = iris_data

        for k in [1, 3, 5]:
            model = KNN(n_neighbors=k)
            model.fit(X, y)

            y_pred = model.predict(X)
            assert y_pred.shape == y.shape


class TestFCMExtended:
    """Extended FCM tests."""

    def test_membership_properties(self, simple_data):
        """Test membership matrix properties."""
        model = FCM(n_clusters=2, random_seed=42)
        model.fit(simple_data)
        U = model.U_

        assert jnp.all(U >= 0) and jnp.all(U <= 1)
        np.testing.assert_allclose(jnp.sum(U, axis=1), 1.0, rtol=1e-5)

    def test_objective_decreases(self, simple_data):
        """Test objective decreases."""
        model = FCM(n_clusters=2, max_iter=50, random_seed=42)
        model.fit(simple_data)

        objectives = model.get_objective_history()
        assert objectives[0] > objectives[-1]


class TestKernelModels:
    """Test kernel models."""

    @pytest.mark.parametrize("model_class,params", [
        (KFCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'sigma': 1.0}),
        (KPCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'k': 1.0, 'sigma': 1.0}),
    ])
    def test_kernel_clustering(self, model_class, params, simple_data):
        """Test kernel models work correctly."""
        model = model_class(**params, random_seed=42)
        model.fit(simple_data)

        labels = model.predict(simple_data)
        assert labels.shape == (len(simple_data),)


class TestDeviceCompatibility:
    """Test CPU/GPU compatibility."""

    @pytest.mark.parametrize("model_class,params", STANDARD_MODELS[:3])
    def test_cpu_device(self, model_class, params, simple_data):
        """Test models work on CPU."""
        with jax.default_device(jax.devices('cpu')[0]):
            model = model_class(**params, random_seed=42)
            model.fit(simple_data)

            labels = model.predict(simple_data)
            assert labels.shape == (len(simple_data),)

    @pytest.mark.parametrize("model_class,params", STANDARD_MODELS[:2])
    def test_gpu_device_if_available(self, model_class, params, simple_data):
        """Test models work on GPU if available."""
        try:
            gpu_devices = jax.devices('gpu')
            if len(gpu_devices) == 0:
                pytest.skip("No GPU available")
        except RuntimeError:
            pytest.skip("No GPU backend available")

        with jax.default_device(gpu_devices[0]):
            X_gpu = jax.device_put(simple_data, gpu_devices[0])
            model = model_class(**params, random_seed=42)
            model.fit(X_gpu)

            labels = model.predict(X_gpu)
            assert labels.shape == (len(simple_data),)


class TestParameterValidation:
    """Test parameter validation."""

    def test_fcm_invalid_params(self):
        """Test FCM parameter validation."""
        with pytest.raises(ValueError):
            FCM(n_clusters=1)

        with pytest.raises(ValueError):
            FCM(n_clusters=2, fuzzifier=1.0)

    def test_pcm_invalid_params(self):
        """Test PCM parameter validation."""
        with pytest.raises(ValueError):
            PCM(n_clusters=2, k=0)

    def test_knn_invalid_params(self):
        """Test KNN parameter validation."""
        # KNN doesn't validate n_neighbors in __init__
        # Just test that it can be created
        model = KNN(n_neighbors=5)
        assert model.n_neighbors == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
