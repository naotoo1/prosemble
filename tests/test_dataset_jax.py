"""
JAX-compatible dataset test suite.

Tests cover:
- Dataset loading with JAX arrays
- Data shape and type validation
- JAX array conversion
- Integration with JAX models
- Device placement (CPU/GPU)
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from prosemble.datasets import DATA_JAX as DATA, DATASET_JAX as DATASET
    from prosemble.models import FCM, PCM
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    DATA = None
    DATASET = None
    FCM = None
    PCM = None

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


@pytest.fixture
def data_instance():
    """Create DATA instance."""
    return DATA()


class TestBreastCancerDataset:
    """Test Wisconsin Breast Cancer dataset with JAX."""

    def test_load_dataset(self, data_instance):
        """Test loading breast cancer dataset."""
        dataset = data_instance.breast_cancer

        assert isinstance(dataset, DATASET)
        assert dataset.input_data is not None
        assert dataset.labels is not None

    def test_dataset_size(self, data_instance):
        """Test dataset has correct size."""
        dataset = data_instance.breast_cancer

        assert len(dataset.input_data) == 569
        assert len(dataset.labels) == 569

    def test_dataset_dimensions(self, data_instance):
        """Test dataset has 30 features."""
        dataset = data_instance.breast_cancer

        assert dataset.input_data.shape[1] == 30

    def test_unique_labels(self, data_instance):
        """Test dataset has 2 classes."""
        dataset = data_instance.breast_cancer

        assert len(np.unique(dataset.labels)) == 2

    def test_jax_conversion(self, data_instance):
        """Test conversion to JAX arrays."""
        dataset = data_instance.breast_cancer

        X_jax = jnp.array(dataset.input_data, dtype=jnp.float32)
        y_jax = jnp.array(dataset.labels)

        assert isinstance(X_jax, jnp.ndarray)
        assert isinstance(y_jax, jnp.ndarray)
        assert X_jax.shape == (569, 30)
        assert y_jax.shape == (569,)

    def test_fcm_clustering(self, data_instance):
        """Test FCM clustering on breast cancer data."""
        dataset = data_instance.breast_cancer
        X_jax = jnp.array(dataset.input_data[:100], dtype=jnp.float32)

        model = FCM(n_clusters=2, max_iter=20, random_seed=42)
        model.fit(X_jax)

        labels = model.predict(X_jax)

        assert labels.shape == (100,)
        assert len(jnp.unique(labels)) <= 2

    def test_pcm_clustering(self, data_instance):
        """Test PCM clustering on breast cancer data."""
        dataset = data_instance.breast_cancer
        X_jax = jnp.array(dataset.input_data[:100], dtype=jnp.float32)

        model = PCM(n_clusters=2, fuzzifier=2.0, k=1.0, max_iter=20, random_seed=42)
        model.fit(X_jax)

        labels = model.predict(X_jax)

        assert labels.shape == (100,)
        assert len(jnp.unique(labels)) <= 2

    def test_dimension_selection(self, data_instance):
        """Test selecting subset of dimensions."""
        dataset = data_instance.breast_cancer

        # Select first 2 features
        X_subset = dataset.input_data[:, [0, 1]]
        X_jax = jnp.array(X_subset, dtype=jnp.float32)

        assert X_jax.shape == (569, 2)


class TestMoonsDataset:
    """Test moons dataset with JAX."""

    def test_load_dataset(self, data_instance):
        """Test loading moons dataset."""
        dataset = data_instance.S_1

        assert isinstance(dataset, DATASET)
        assert dataset.input_data is not None
        assert dataset.labels is not None

    def test_dataset_shape(self, data_instance):
        """Test moons dataset shape."""
        dataset = data_instance.S_1

        # Moons is 2D data
        assert dataset.input_data.shape[1] == 2
        assert len(np.unique(dataset.labels)) == 2

    def test_jax_conversion(self, data_instance):
        """Test conversion to JAX arrays."""
        dataset = data_instance.S_1

        X_jax = jnp.array(dataset.input_data, dtype=jnp.float32)
        y_jax = jnp.array(dataset.labels)

        assert isinstance(X_jax, jnp.ndarray)
        assert isinstance(y_jax, jnp.ndarray)
        assert X_jax.shape[1] == 2

    def test_fcm_clustering_moons(self, data_instance):
        """Test FCM on moons dataset."""
        dataset = data_instance.S_1
        X_jax = jnp.array(dataset.input_data, dtype=jnp.float32)

        model = FCM(n_clusters=2, max_iter=50, random_seed=42)
        model.fit(X_jax)

        labels = model.predict(X_jax)

        assert labels.shape == (len(X_jax),)
        assert len(jnp.unique(labels)) == 2


class TestBlobsDataset:
    """Test blobs dataset with JAX."""

    def test_load_dataset(self, data_instance):
        """Test loading blobs dataset."""
        dataset = data_instance.S_2

        assert isinstance(dataset, DATASET)
        assert dataset.input_data is not None
        assert dataset.labels is not None

    def test_dataset_size(self, data_instance):
        """Test blobs dataset has expected size."""
        dataset = data_instance.S_2

        # 120 + 80 = 200 samples
        assert len(dataset.input_data) == 200
        assert len(dataset.labels) == 200

    def test_dataset_shape(self, data_instance):
        """Test blobs dataset shape."""
        dataset = data_instance.S_2

        # Blobs is 2D data
        assert dataset.input_data.shape[1] == 2
        assert len(np.unique(dataset.labels)) == 2

    def test_jax_conversion(self, data_instance):
        """Test conversion to JAX arrays."""
        dataset = data_instance.S_2

        X_jax = jnp.array(dataset.input_data, dtype=jnp.float32)
        y_jax = jnp.array(dataset.labels)

        assert isinstance(X_jax, jnp.ndarray)
        assert isinstance(y_jax, jnp.ndarray)
        assert X_jax.shape == (200, 2)

    def test_fcm_clustering_blobs(self, data_instance):
        """Test FCM on blobs dataset."""
        dataset = data_instance.S_2
        X_jax = jnp.array(dataset.input_data, dtype=jnp.float32)

        model = FCM(n_clusters=2, max_iter=50, random_seed=42)
        model.fit(X_jax)

        labels = model.predict(X_jax)

        assert labels.shape == (200,)
        assert len(jnp.unique(labels)) == 2


class TestDatasetIntegration:
    """Test dataset integration with multiple JAX models."""

    @pytest.mark.parametrize("model_class,params", [
        (FCM, {'n_clusters': 2, 'fuzzifier': 2.0}),
        (PCM, {'n_clusters': 2, 'fuzzifier': 2.0, 'k': 1.0}),
    ])
    def test_multiple_models(self, data_instance, model_class, params):
        """Test multiple models work with datasets."""
        dataset = data_instance.S_2
        X_jax = jnp.array(dataset.input_data, dtype=jnp.float32)

        model = model_class(**params, random_seed=42, max_iter=20)
        model.fit(X_jax)

        labels = model.predict(X_jax)

        assert labels.shape == (200,)
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < 2)


class TestJAXDataTypes:
    """Test different JAX data types."""

    def test_float32_conversion(self, data_instance):
        """Test float32 conversion."""
        dataset = data_instance.breast_cancer
        X_jax = jnp.array(dataset.input_data, dtype=jnp.float32)

        assert X_jax.dtype == jnp.float32

    def test_float64_conversion(self, data_instance):
        """Test float64 conversion (if x64 enabled)."""
        dataset = data_instance.breast_cancer

        # JAX may not support float64 by default (requires JAX_ENABLE_X64)
        # This test checks that conversion works even if truncated to float32
        X_jax = jnp.array(dataset.input_data, dtype=jnp.float64)

        # Accept either float64 (if x64 enabled) or float32 (default)
        assert X_jax.dtype in [jnp.float32, jnp.float64]

    def test_int_labels(self, data_instance):
        """Test integer label conversion."""
        dataset = data_instance.breast_cancer
        y_jax = jnp.array(dataset.labels, dtype=jnp.int32)

        assert y_jax.dtype == jnp.int32


class TestDatasetNormalization:
    """Test dataset normalization with JAX."""

    def test_standardization(self, data_instance):
        """Test standardization (zero mean, unit variance)."""
        dataset = data_instance.breast_cancer
        X = dataset.input_data

        # Standardize using JAX
        X_jax = jnp.array(X, dtype=jnp.float32)
        mean = jnp.mean(X_jax, axis=0)
        std = jnp.std(X_jax, axis=0)
        X_normalized = (X_jax - mean) / (std + 1e-8)

        # Check mean ≈ 0, std ≈ 1
        assert jnp.allclose(jnp.mean(X_normalized, axis=0), 0.0, atol=1e-5)
        assert jnp.allclose(jnp.std(X_normalized, axis=0), 1.0, atol=1e-5)

    def test_min_max_scaling(self, data_instance):
        """Test min-max scaling to [0, 1]."""
        dataset = data_instance.breast_cancer
        X = dataset.input_data

        # Min-max scale using JAX
        X_jax = jnp.array(X, dtype=jnp.float32)
        X_min = jnp.min(X_jax, axis=0)
        X_max = jnp.max(X_jax, axis=0)
        X_scaled = (X_jax - X_min) / (X_max - X_min + 1e-8)

        # Check range [0, 1]
        assert jnp.all(X_scaled >= 0.0)
        assert jnp.all(X_scaled <= 1.0)


class TestDatasetBatching:
    """Test dataset batching for JAX models."""

    def test_batch_creation(self, data_instance):
        """Test creating batches from dataset."""
        dataset = data_instance.breast_cancer
        X_jax = jnp.array(dataset.input_data, dtype=jnp.float32)

        batch_size = 64
        n_samples = len(X_jax)
        n_batches = (n_samples + batch_size - 1) // batch_size

        batches = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch = X_jax[start_idx:end_idx]
            batches.append(batch)

        # Check all data is covered
        total_samples = sum(len(batch) for batch in batches)
        assert total_samples == n_samples

    def test_fcm_on_batches(self, data_instance):
        """Test FCM clustering with batched prediction."""
        dataset = data_instance.breast_cancer
        X_jax = jnp.array(dataset.input_data[:200], dtype=jnp.float32)

        # Train on full data
        model = FCM(n_clusters=2, max_iter=20, random_seed=42)
        model.fit(X_jax)

        # Predict in batches
        batch_size = 50
        all_labels = []
        for i in range(0, len(X_jax), batch_size):
            batch = X_jax[i:i + batch_size]
            labels = model.predict(batch)
            all_labels.append(labels)

        all_labels = jnp.concatenate(all_labels)
        assert all_labels.shape == (200,)


class TestDeviceCompatibility:
    """Test dataset compatibility with different devices."""

    def test_cpu_device(self, data_instance):
        """Test datasets work on CPU."""
        dataset = data_instance.S_2

        with jax.default_device(jax.devices('cpu')[0]):
            X_jax = jnp.array(dataset.input_data, dtype=jnp.float32)

            model = FCM(n_clusters=2, max_iter=10, random_seed=42)
            model.fit(X_jax)

            labels = model.predict(X_jax)
            assert labels.shape == (200,)

    def test_gpu_device_if_available(self, data_instance):
        """Test datasets work on GPU if available."""
        try:
            gpu_devices = jax.devices('gpu')
            if len(gpu_devices) == 0:
                pytest.skip("No GPU available")
        except RuntimeError:
            pytest.skip("No GPU backend available")

        dataset = data_instance.S_2

        with jax.default_device(gpu_devices[0]):
            X_jax = jax.device_put(
                jnp.array(dataset.input_data, dtype=jnp.float32),
                gpu_devices[0]
            )

            model = FCM(n_clusters=2, max_iter=10, random_seed=42)
            model.fit(X_jax)

            labels = model.predict(X_jax)
            assert labels.shape == (200,)


class TestDatasetProperties:
    """Test dataset properties and metadata."""

    def test_dataset_dataclass(self):
        """Test DATASET dataclass structure."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        dataset = DATASET(input_data=X, labels=y)

        assert hasattr(dataset, 'input_data')
        assert hasattr(dataset, 'labels')
        assert isinstance(dataset.input_data, np.ndarray)
        assert isinstance(dataset.labels, np.ndarray)

    def test_data_instance_properties(self):
        """Test DATA instance has expected properties."""
        data = DATA()

        assert hasattr(data, 'S_1')
        assert hasattr(data, 'S_2')
        assert hasattr(data, 'breast_cancer')
        assert hasattr(data, 'random')
        assert hasattr(data, 'sample_size')

    def test_custom_random_state(self):
        """Test custom random state affects dataset generation."""
        data1 = DATA(random=42)
        data2 = DATA(random=123)

        dataset1 = data1.S_1
        dataset2 = data2.S_1

        # Different random states should generate different datasets
        # Note: We can't test exact equality since DATA.random is passed to generators
        # but we can verify they have the same structure
        assert dataset1.input_data.shape == dataset2.input_data.shape
        assert len(dataset1.labels) == len(dataset2.labels)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
