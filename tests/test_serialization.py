"""Tests for prosemble.core.serialization — unified save/load."""

import os
import tempfile

import pytest
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from prosemble.models import GLVQ, NeuralGas, FCM


@pytest.fixture
def iris_data():
    """Small synthetic classification dataset."""
    np.random.seed(42)
    X = np.random.randn(30, 4).astype(np.float32)
    y = np.array([0]*10 + [1]*10 + [2]*10)
    return jnp.array(X), jnp.array(y)


@pytest.fixture
def cluster_data():
    """Small synthetic clustering dataset."""
    np.random.seed(42)
    X = np.random.randn(30, 4).astype(np.float32)
    return jnp.array(X)


class TestSupervisedSaveLoad:

    def test_glvq_roundtrip(self, iris_data, tmp_path):
        X, y = iris_data
        model = GLVQ(n_prototypes_per_class=1, max_iter=3, random_seed=42)
        model.fit(X, y)
        preds_before = model.predict(X)

        path = str(tmp_path / "glvq.npz")
        model.save(path)

        loaded = GLVQ.load(path)
        preds_after = loaded.predict(X)

        npt.assert_array_equal(preds_before, preds_after)
        npt.assert_array_almost_equal(
            np.asarray(model.prototypes_),
            np.asarray(loaded.prototypes_),
        )

    def test_metadata_preserved(self, iris_data, tmp_path):
        X, y = iris_data
        model = GLVQ(n_prototypes_per_class=1, max_iter=3, random_seed=42)
        model.fit(X, y)

        path = str(tmp_path / "glvq_meta.npz")
        model.save(path)
        loaded = GLVQ.load(path)

        assert loaded.n_iter_ is not None
        assert loaded.loss_ is not None
        assert loaded.n_classes_ == model.n_classes_

    def test_quantized_save(self, iris_data, tmp_path):
        X, y = iris_data
        model = GLVQ(n_prototypes_per_class=1, max_iter=3, random_seed=42)
        model.fit(X, y)

        path = str(tmp_path / "glvq_q.npz")
        model.save(path, quantize='float16')

        # Model in memory should be unchanged
        assert model.prototypes_.dtype == jnp.float32

        loaded = GLVQ.load(path)
        assert loaded._quantized_dtype == 'float16'


class TestUnsupervisedSaveLoad:

    def test_neural_gas_roundtrip(self, cluster_data, tmp_path):
        X = cluster_data
        model = NeuralGas(n_prototypes=3, max_iter=3, random_seed=42)
        model.fit(X)
        preds_before = model.predict(X)

        path = str(tmp_path / "ng.npz")
        model.save(path)

        loaded = NeuralGas.load(path)
        preds_after = loaded.predict(X)

        npt.assert_array_equal(preds_before, preds_after)

    def test_metadata_preserved(self, cluster_data, tmp_path):
        X = cluster_data
        model = NeuralGas(n_prototypes=3, max_iter=3, random_seed=42)
        model.fit(X)

        path = str(tmp_path / "ng_meta.npz")
        model.save(path)
        loaded = NeuralGas.load(path)

        assert loaded.n_iter_ is not None
        assert loaded.loss_ is not None


class TestFuzzySaveLoad:

    def test_fcm_roundtrip(self, cluster_data, tmp_path):
        X = cluster_data
        model = FCM(n_clusters=3, max_iter=10, random_seed=42)
        model.fit(X)
        preds_before = model.predict(X)

        path = str(tmp_path / "fcm.npz")
        model.save(path)

        loaded = FCM.load(path)
        preds_after = loaded.predict(X)

        npt.assert_array_equal(preds_before, preds_after)

    def test_metadata_preserved(self, cluster_data, tmp_path):
        X = cluster_data
        model = FCM(n_clusters=3, max_iter=10, random_seed=42)
        model.fit(X)

        path = str(tmp_path / "fcm_meta.npz")
        model.save(path)
        loaded = FCM.load(path)

        assert loaded.n_iter_ is not None
        assert loaded.objective_ is not None


class TestSchemaVersion:

    def test_schema_version_in_metadata(self, iris_data, tmp_path):
        import json
        X, y = iris_data
        model = GLVQ(n_prototypes_per_class=1, max_iter=3, random_seed=42)
        model.fit(X, y)

        path = str(tmp_path / "versioned.npz")
        model.save(path)

        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data['__metadata__']))
        assert metadata['schema_version'] == 1


class TestRegistryLoad:

    def test_load_unknown_class_raises(self, tmp_path):
        """Load should raise for unknown model class."""
        import json
        path = str(tmp_path / "bad.npz")
        np.savez_compressed(
            path,
            __metadata__=np.array(json.dumps({
                'class_name': 'NonExistentModel',
                'module': 'fake',
                'hyperparams': {},
                'fitted_array_names': [],
                'quantized_dtype': None,
            })),
        )
        with pytest.raises(ValueError, match="Unknown model class"):
            GLVQ.load(path)
