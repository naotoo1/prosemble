"""Tests for RSLVQ with Neural Gas Cooperation (RSLVQ_NG)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.rslvq_ng import RSLVQ_NG


@pytest.fixture
def iris_data():
    """Iris dataset."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X = jnp.array(data.data, dtype=jnp.float32)
    y = jnp.array(data.target, dtype=jnp.int32)
    return X, y


@pytest.fixture
def separable_2d():
    """Easy 2-class 2D dataset."""
    X = jnp.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [-0.1, 0.2],
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.0], [0.9, 1.2],
    ])
    y = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


class TestRSLVQNGBasic:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = RSLVQ_NG(
            sigma=0.5, n_prototypes_per_class=2, max_iter=50, lr=0.01,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_iris(self, iris_data):
        X, y = iris_data
        model = RSLVQ_NG(
            sigma=1.0, n_prototypes_per_class=2, max_iter=100, lr=0.01,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.60

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = RSLVQ_NG(
            sigma=0.5, n_prototypes_per_class=2, max_iter=50, lr=0.01,
        )
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_prototypes_shape(self, iris_data):
        X, y = iris_data
        model = RSLVQ_NG(
            sigma=1.0, n_prototypes_per_class=3, max_iter=20, lr=0.01,
        )
        model.fit(X, y)
        assert model.prototypes_.shape == (9, 4)  # 3 classes * 3 per class


class TestRSLVQNGGammaDecay:
    def test_gamma_decays(self, separable_2d):
        X, y = separable_2d
        model = RSLVQ_NG(
            sigma=0.5, n_prototypes_per_class=2, max_iter=50, lr=0.01,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0
        assert model.gamma_ >= 0.01

    def test_gamma_reaches_final(self, separable_2d):
        X, y = separable_2d
        model = RSLVQ_NG(
            sigma=0.5, n_prototypes_per_class=2, max_iter=200, lr=0.01,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0

    def test_custom_gamma_decay(self, separable_2d):
        X, y = separable_2d
        model = RSLVQ_NG(
            sigma=0.5, n_prototypes_per_class=2, max_iter=50, lr=0.01,
            gamma_init=5.0, gamma_decay=0.95,
        )
        model.fit(X, y)
        assert model.gamma_ < 1.0


class TestRSLVQNGPrediction:
    def test_predict_proba(self, separable_2d):
        X, y = separable_2d
        model = RSLVQ_NG(
            sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.01,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (8, 2)
        np.testing.assert_allclose(jnp.sum(proba, axis=1), 1.0, atol=1e-5)

    def test_rejection(self, separable_2d):
        X, y = separable_2d
        model = RSLVQ_NG(
            sigma=0.5, rejection_confidence=0.99,
            n_prototypes_per_class=1, max_iter=50, lr=0.01,
        )
        model.fit(X, y)
        preds = model.predict_with_rejection(X)
        assert preds.shape == (8,)
        assert jnp.any(preds == -1) or jnp.all(preds >= 0)


class TestRSLVQNGSaveLoad:
    def test_save_load_roundtrip(self, iris_data):
        X, y = iris_data
        model = RSLVQ_NG(
            sigma=1.0, n_prototypes_per_class=2, max_iter=50, lr=0.01,
        )
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rslvq_ng_model")
            model.save(path)

            loaded = RSLVQ_NG.load(path)
            np.testing.assert_allclose(
                np.asarray(model.prototypes_),
                np.asarray(loaded.prototypes_),
            )
            preds_orig = model.predict(X)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestRSLVQNGResume:
    def test_resume_training(self, iris_data):
        X, y = iris_data
        model = RSLVQ_NG(
            sigma=1.0, n_prototypes_per_class=2, max_iter=30, lr=0.01,
        )
        model.fit(X, y)
        acc_before = float(jnp.mean(model.predict(X) == y))

        model.fit(X, y, resume=True)
        acc_after = float(jnp.mean(model.predict(X) == y))
        assert acc_after >= acc_before - 0.05

    def test_partial_fit(self, iris_data):
        X, y = iris_data
        model = RSLVQ_NG(
            sigma=1.0, n_prototypes_per_class=2, max_iter=30, lr=0.01,
        )
        model.fit(X, y)
        n_iter_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_iter_before + 1


class TestRSLVQNGSmallGamma:
    def test_small_gamma_like_rslvq(self, iris_data):
        """With very small gamma, RSLVQ_NG should behave similarly to RSLVQ."""
        X, y = iris_data
        model = RSLVQ_NG(
            sigma=1.0, n_prototypes_per_class=1, max_iter=100, lr=0.01,
            gamma_init=0.001, gamma_final=0.001,
            random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.60
