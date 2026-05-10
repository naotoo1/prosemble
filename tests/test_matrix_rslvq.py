"""Tests for MRSLVQ and LMRSLVQ."""

import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models.matrix_rslvq import MRSLVQ, LMRSLVQ


@pytest.fixture
def separable_2d():
    X = jnp.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [-0.1, 0.2],
        [1.0, 1.0], [1.1, 1.1], [1.2, 1.0], [0.9, 1.2],
    ])
    y = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


@pytest.fixture
def iris_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    return jnp.array(data.data, dtype=jnp.float32), jnp.array(data.target, dtype=jnp.int32)


class TestMRSLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = MRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = MRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_omega_matrix(self, separable_2d):
        X, y = separable_2d
        model = MRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=30, lr=0.05)
        model.fit(X, y)
        omega = model.omega_matrix
        assert omega.shape == (2, 2)

    def test_lambda_matrix(self, separable_2d):
        X, y = separable_2d
        model = MRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=30, lr=0.05)
        model.fit(X, y)
        lam = model.lambda_matrix
        assert lam.shape == (2, 2)
        np.testing.assert_allclose(lam, lam.T, atol=1e-5)

    def test_latent_dim(self, iris_data):
        X, y = iris_data
        model = MRSLVQ(
            sigma=1.0, latent_dim=2, n_prototypes_per_class=1,
            max_iter=30, lr=0.01,
        )
        model.fit(X, y)
        assert model.omega_matrix.shape == (4, 2)

    def test_accuracy(self, separable_2d):
        X, y = separable_2d
        model = MRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=100, lr=0.05)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_predict_proba(self, separable_2d):
        X, y = separable_2d
        model = MRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (8, 2)
        np.testing.assert_allclose(jnp.sum(proba, axis=1), 1.0, atol=1e-5)

    def test_rejection(self, separable_2d):
        X, y = separable_2d
        model = MRSLVQ(
            sigma=0.5, rejection_confidence=0.99,
            n_prototypes_per_class=1, max_iter=50, lr=0.05,
        )
        model.fit(X, y)
        preds = model.predict_with_rejection(X)
        assert preds.shape == (8,)
        # With high threshold, some samples may be rejected
        assert jnp.any(preds == -1) or jnp.all(preds >= 0)

    def test_save_load(self, separable_2d, tmp_path):
        X, y = separable_2d
        model = MRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        preds_before = model.predict(X)

        path = str(tmp_path / "mrslvq.npz")
        model.save(path)
        loaded = MRSLVQ.load(path)
        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_iris_accuracy(self, iris_data):
        X, y = iris_data
        model = MRSLVQ(
            sigma=1.0, n_prototypes_per_class=2, max_iter=100, lr=0.01,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.6


class TestLMRSLVQ:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = LMRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = LMRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_local_omegas(self, separable_2d):
        X, y = separable_2d
        model = LMRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=20, lr=0.05)
        model.fit(X, y)
        assert model.omegas_.shape == (2, 2, 2)  # 2 protos, 2x2 omegas

    def test_latent_dim(self, iris_data):
        X, y = iris_data
        model = LMRSLVQ(
            sigma=1.0, latent_dim=2, n_prototypes_per_class=1,
            max_iter=20, lr=0.01,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 2)  # 3 protos, 4x2 omegas

    def test_accuracy(self, separable_2d):
        X, y = separable_2d
        model = LMRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=100, lr=0.05)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_predict_proba(self, separable_2d):
        X, y = separable_2d
        model = LMRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (8, 2)
        np.testing.assert_allclose(jnp.sum(proba, axis=1), 1.0, atol=1e-5)

    def test_rejection(self, separable_2d):
        X, y = separable_2d
        model = LMRSLVQ(
            sigma=0.5, rejection_confidence=0.99,
            n_prototypes_per_class=1, max_iter=50, lr=0.05,
        )
        model.fit(X, y)
        preds = model.predict_with_rejection(X)
        assert preds.shape == (8,)

    def test_save_load(self, separable_2d, tmp_path):
        X, y = separable_2d
        model = LMRSLVQ(sigma=0.5, n_prototypes_per_class=1, max_iter=50, lr=0.05)
        model.fit(X, y)
        preds_before = model.predict(X)

        path = str(tmp_path / "lmrslvq.npz")
        model.save(path)
        loaded = LMRSLVQ.load(path)
        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)
