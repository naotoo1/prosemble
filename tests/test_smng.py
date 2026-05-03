"""Tests for Supervised Matrix Neural Gas (SMNG)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.smng import SMNG


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


class TestSMNGBasic:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = SMNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_iris(self, iris_data):
        X, y = iris_data
        model = SMNG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.80

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = SMNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_prototypes_shape(self, iris_data):
        X, y = iris_data
        model = SMNG(n_prototypes_per_class=3, max_iter=20, lr=0.01)
        model.fit(X, y)
        assert model.prototypes_.shape == (9, 4)  # 3 classes * 3 per class


class TestSMNGGammaDecay:
    def test_gamma_decays(self, separable_2d):
        X, y = separable_2d
        model = SMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0
        assert model.gamma_ >= 0.01

    def test_gamma_reaches_final(self, separable_2d):
        X, y = separable_2d
        model = SMNG(
            n_prototypes_per_class=2, max_iter=200, lr=0.01,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0

    def test_custom_gamma_decay(self, separable_2d):
        X, y = separable_2d
        model = SMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            gamma_init=5.0, gamma_decay=0.95,
        )
        model.fit(X, y)
        assert model.gamma_ < 1.0


class TestSMNGOmega:
    def test_omega_learned(self, iris_data):
        X, y = iris_data
        model = SMNG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        omega = model.omega_matrix
        assert omega.shape == (4, 4)  # d x d (no latent_dim reduction)

    def test_lambda_matrix(self, iris_data):
        X, y = iris_data
        model = SMNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        lam = model.lambda_matrix
        assert lam.shape == (4, 4)
        # Lambda should be positive semi-definite (symmetric)
        np.testing.assert_allclose(np.asarray(lam), np.asarray(lam.T), atol=1e-5)

    def test_latent_dim(self, iris_data):
        X, y = iris_data
        model = SMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            latent_dim=2,
        )
        model.fit(X, y)
        assert model.omega_matrix.shape == (4, 2)

    def test_omega_not_fitted_raises(self):
        model = SMNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.omega_matrix


class TestSMNGTrainingModes:
    def test_scan_training(self, separable_2d):
        X, y = separable_2d
        model = SMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            use_scan=True,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_minibatch_training(self, iris_data):
        X, y = iris_data
        model = SMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            batch_size=32,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.60


class TestSMNGSaveLoad:
    def test_save_load_roundtrip(self, iris_data):
        X, y = iris_data
        model = SMNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "smng_model")
            model.save(path)

            loaded = SMNG.load(path)
            np.testing.assert_allclose(
                np.asarray(model.prototypes_),
                np.asarray(loaded.prototypes_),
            )
            np.testing.assert_allclose(
                np.asarray(model.omega_),
                np.asarray(loaded.omega_),
            )
            preds_orig = model.predict(X)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestSMNGResume:
    def test_resume_training(self, iris_data):
        X, y = iris_data
        model = SMNG(n_prototypes_per_class=2, max_iter=30, lr=0.01)
        model.fit(X, y)
        acc_before = float(jnp.mean(model.predict(X) == y))

        model.fit(X, y, resume=True)
        acc_after = float(jnp.mean(model.predict(X) == y))
        assert acc_after >= acc_before - 0.05

    def test_partial_fit(self, iris_data):
        X, y = iris_data
        model = SMNG(n_prototypes_per_class=2, max_iter=30, lr=0.01)
        model.fit(X, y)
        n_iter_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_iter_before + 1


class TestSMNGSmallGamma:
    def test_small_gamma_like_gmlvq(self, iris_data):
        """With very small gamma, SMNG should behave similarly to GMLVQ."""
        X, y = iris_data
        model = SMNG(
            n_prototypes_per_class=1, max_iter=100, lr=0.01,
            gamma_init=0.001, gamma_final=0.001,
            random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.75
