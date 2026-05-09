"""Tests for Tangent Cross-Entropy LVQ with Neural Gas (TCELVQ-NG)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.tcelvq_ng import TCELVQ_NG


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


# ============================================================
# Basic tests
# ============================================================

class TestTCELVQNGBasic:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            subspace_dim=1,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_iris(self, iris_data):
        X, y = iris_data
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=100, lr=0.01,
            subspace_dim=2,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.80

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            subspace_dim=1,
        )
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_prototypes_shape(self, iris_data):
        X, y = iris_data
        model = TCELVQ_NG(
            n_prototypes_per_class=3, max_iter=20, lr=0.01,
            subspace_dim=2,
        )
        model.fit(X, y)
        assert model.prototypes_.shape == (9, 4)  # 3 classes * 3 per class


# ============================================================
# Tangent subspace tests
# ============================================================

class TestTCELVQNGOmegas:
    def test_omegas_shape(self, iris_data):
        X, y = iris_data
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            subspace_dim=2,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (6, 4, 2)  # 6 protos, 4x2 each

    def test_omegas_orthogonal(self, iris_data):
        """Tangent bases should remain orthonormal after training."""
        X, y = iris_data
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            subspace_dim=2,
        )
        model.fit(X, y)
        for k in range(model.omegas_.shape[0]):
            omega_k = model.omegas_[k]  # (d, s)
            gram = omega_k.T @ omega_k  # (s, s) should be ~identity
            np.testing.assert_allclose(
                np.asarray(gram), np.eye(2), atol=1e-4,
                err_msg=f"Omega_{k} not orthonormal"
            )

    def test_different_subspace_dims(self, iris_data):
        X, y = iris_data
        for sdim in [1, 2, 3]:
            model = TCELVQ_NG(
                n_prototypes_per_class=1, max_iter=20, lr=0.01,
                subspace_dim=sdim,
            )
            model.fit(X, y)
            assert model.omegas_.shape == (3, 4, sdim)


# ============================================================
# Gamma decay tests
# ============================================================

class TestTCELVQNGGammaDecay:
    def test_gamma_decays(self, separable_2d):
        X, y = separable_2d
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            subspace_dim=1, gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0
        assert model.gamma_ >= 0.01 - 1e-6  # float32 precision

    def test_custom_gamma_decay(self, separable_2d):
        X, y = separable_2d
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            subspace_dim=1, gamma_init=5.0, gamma_final=0.01,
            gamma_decay=0.95,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0


# ============================================================
# Probability tests
# ============================================================

class TestTCELVQNGProbabilities:
    def test_predict_proba_sums_to_one(self, iris_data):
        X, y = iris_data
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            subspace_dim=2,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        row_sums = jnp.sum(proba, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_proba_shape(self, iris_data):
        X, y = iris_data
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            subspace_dim=2,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (150, 3)

    def test_predict_proba_nonnegative(self, iris_data):
        X, y = iris_data
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            subspace_dim=2,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert jnp.all(proba >= 0)


# ============================================================
# Save/load tests
# ============================================================

class TestTCELVQNGSaveLoad:
    def test_save_load_roundtrip(self, iris_data):
        X, y = iris_data
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            subspace_dim=2,
        )
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'tcelvq_ng.npz')
            model.save(path)

            loaded = TCELVQ_NG.load(path)
            np.testing.assert_allclose(
                np.asarray(model.prototypes_),
                np.asarray(loaded.prototypes_),
            )
            np.testing.assert_allclose(
                np.asarray(model.omegas_),
                np.asarray(loaded.omegas_),
            )
            preds_orig = model.predict(X)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_orig, preds_loaded)
            assert loaded.gamma_ == pytest.approx(model.gamma_)


# ============================================================
# Resume tests
# ============================================================

class TestTCELVQNGResume:
    def test_resume_training(self, iris_data):
        X, y = iris_data
        model = TCELVQ_NG(
            n_prototypes_per_class=2, max_iter=30, lr=0.01,
            subspace_dim=2,
        )
        model.fit(X, y)
        loss_after_30 = model.loss_

        model.max_iter = 30
        model.fit(X, y, resume=True)
        loss_after_60 = model.loss_

        assert loss_after_60 <= loss_after_30 + 0.1  # should not get worse
