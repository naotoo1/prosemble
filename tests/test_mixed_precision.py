"""Tests for mixed precision training."""

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models.glvq import GLVQ
from prosemble.models.matrix_lvq import GMLVQ


@pytest.fixture
def separable_2d():
    """Easy 2-class 2D dataset."""
    X = jnp.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [-0.1, 0.2],
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.0], [0.9, 1.2],
    ])
    y = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


@pytest.fixture
def iris_subset():
    """3-class dataset from Iris."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X = jnp.array(data.data, dtype=jnp.float32)
    y = jnp.array(data.target, dtype=jnp.int32)
    return X, y


class TestMixedPrecisionBFloat16:
    def test_glvq_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=50, lr=0.1,
            mixed_precision='bfloat16',
        )
        model.fit(X, y)
        acc = float(jnp.mean(model.predict(X) == y))
        assert acc >= 0.75

    def test_master_weights_stay_float32(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.1,
            mixed_precision='bfloat16',
        )
        model.fit(X, y)
        assert model.prototypes_.dtype == jnp.float32

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=50, lr=0.1,
            mixed_precision='bfloat16', use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_gmlvq_multi_key_params(self, iris_subset):
        X, y = iris_subset
        model = GMLVQ(
            n_prototypes_per_class=1, max_iter=30, lr=0.01,
            mixed_precision='bfloat16',
        )
        model.fit(X, y)
        assert model.omega_.dtype == jnp.float32
        acc = float(jnp.mean(model.predict(X) == y))
        assert acc >= 0.7


class TestMixedPrecisionFloat16:
    def test_glvq_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=50, lr=0.1,
            mixed_precision='float16',
        )
        model.fit(X, y)
        acc = float(jnp.mean(model.predict(X) == y))
        assert acc >= 0.75

    def test_master_weights_stay_float32(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.1,
            mixed_precision='float16',
        )
        model.fit(X, y)
        assert model.prototypes_.dtype == jnp.float32


class TestTrainingPaths:
    """Verify mixed precision works across all training paths."""

    def test_scan_path(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=30, lr=0.1,
            mixed_precision='bfloat16', use_scan=True,
        )
        model.fit(X, y)
        assert model.loss_ < float('inf')

    def test_python_loop_path(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=30, lr=0.1,
            mixed_precision='bfloat16', use_scan=False,
        )
        model.fit(X, y)
        assert model.loss_ < float('inf')

    def test_minibatch_scan_path(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=30, lr=0.1,
            mixed_precision='bfloat16', use_scan=True, batch_size=4,
        )
        model.fit(X, y)
        assert model.loss_ < float('inf')

    def test_minibatch_python_path(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=30, lr=0.1,
            mixed_precision='bfloat16', use_scan=False, batch_size=4,
        )
        model.fit(X, y)
        assert model.loss_ < float('inf')


class TestValidation:
    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="mixed_precision"):
            GLVQ(mixed_precision='float64')

    def test_none_is_default(self, separable_2d):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=20, lr=0.1,
            random_seed=42,
        )
        model.fit(X, y)
        loss_default = model.loss_history_

        model2 = GLVQ(
            n_prototypes_per_class=1, max_iter=20, lr=0.1,
            random_seed=42, mixed_precision=None,
        )
        model2.fit(X, y)
        loss_none = model2.loss_history_

        np.testing.assert_array_almost_equal(loss_default, loss_none)


class TestSerialization:
    def test_save_load_roundtrip(self, separable_2d, tmp_path):
        X, y = separable_2d
        model = GLVQ(
            n_prototypes_per_class=1, max_iter=20, lr=0.1,
            mixed_precision='bfloat16',
        )
        model.fit(X, y)
        path = tmp_path / "model_mp.npz"
        model.save(str(path))

        loaded = GLVQ.load(str(path))
        assert loaded.mixed_precision == 'bfloat16'
        np.testing.assert_array_equal(
            np.array(model.predict(X)),
            np.array(loaded.predict(X)),
        )
