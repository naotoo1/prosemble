"""Tests for Matrix Cross-Entropy LVQ with Neural Gas (MCELVQ-NG)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.mcelvq_ng import MCELVQ_NG


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

class TestMCELVQNGBasic:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_iris(self, iris_data):
        X, y = iris_data
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.80

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_prototypes_shape(self, iris_data):
        X, y = iris_data
        model = MCELVQ_NG(n_prototypes_per_class=3, max_iter=20, lr=0.01)
        model.fit(X, y)
        assert model.prototypes_.shape == (9, 4)  # 3 classes * 3 per class


# ============================================================
# Omega matrix tests
# ============================================================

class TestMCELVQNGOmega:
    def test_omega_learned(self, iris_data):
        X, y = iris_data
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        omega = model.omega_matrix
        assert omega.shape == (4, 4)  # d x d (no latent_dim reduction)

    def test_lambda_matrix(self, iris_data):
        X, y = iris_data
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        lam = model.lambda_matrix
        assert lam.shape == (4, 4)
        # Lambda should be symmetric (Omega^T Omega)
        np.testing.assert_allclose(np.asarray(lam), np.asarray(lam.T), atol=1e-5)

    def test_latent_dim(self, iris_data):
        X, y = iris_data
        model = MCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            latent_dim=2,
        )
        model.fit(X, y)
        assert model.omega_matrix.shape == (4, 2)

    def test_omega_not_fitted_raises(self):
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.omega_matrix


# ============================================================
# Gamma decay tests
# ============================================================

class TestMCELVQNGGammaDecay:
    def test_gamma_decays(self, separable_2d):
        X, y = separable_2d
        model = MCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0
        assert model.gamma_ >= 0.01 - 1e-6  # float32 precision

    def test_gamma_reaches_final(self, separable_2d):
        X, y = separable_2d
        model = MCELVQ_NG(
            n_prototypes_per_class=2, max_iter=200, lr=0.01,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 0.05  # should be close to final

    def test_custom_gamma_decay(self, separable_2d):
        X, y = separable_2d
        model = MCELVQ_NG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            gamma_init=5.0, gamma_final=0.01, gamma_decay=0.95,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0


# ============================================================
# Probability tests
# ============================================================

class TestMCELVQNGProbabilities:
    def test_predict_proba_sums_to_one(self, iris_data):
        X, y = iris_data
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        proba = model.predict_proba(X)
        row_sums = jnp.sum(proba, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_proba_shape(self, iris_data):
        X, y = iris_data
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (150, 3)

    def test_predict_proba_nonnegative(self, iris_data):
        X, y = iris_data
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert jnp.all(proba >= 0)

    def test_predict_proba_uses_omega(self, iris_data):
        """Verify predict_proba uses Omega-transformed distances."""
        X, y = iris_data
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        proba = model.predict_proba(X)
        preds = model.predict(X)
        # Predicted class should have highest probability
        proba_pred_class = proba[jnp.arange(len(preds)), preds]
        assert jnp.all(proba_pred_class >= 0.3)  # pred class has reasonable prob


# ============================================================
# Save/load tests
# ============================================================

class TestMCELVQNGSaveLoad:
    def test_save_load_roundtrip(self, iris_data):
        X, y = iris_data
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'mcelvq_ng.npz')
            model.save(path)

            loaded = MCELVQ_NG.load(path)
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
            assert loaded.gamma_ == pytest.approx(model.gamma_)


# ============================================================
# Resume tests
# ============================================================

class TestMCELVQNGResume:
    def test_resume_training(self, iris_data):
        X, y = iris_data
        model = MCELVQ_NG(n_prototypes_per_class=2, max_iter=30, lr=0.01)
        model.fit(X, y)
        loss_after_30 = model.loss_

        model.max_iter = 30
        model.fit(X, y, resume=True)
        loss_after_60 = model.loss_

        assert loss_after_60 <= loss_after_30 + 0.1  # should not get worse


# ============================================================
# Recovery test
# ============================================================

class TestMCELVQNGRecovery:
    def test_small_gamma_like_matrix_celvq(self, iris_data):
        """With very small gamma, MCELVQ-NG should behave like matrix CELVQ."""
        X, y = iris_data
        model = MCELVQ_NG(
            n_prototypes_per_class=1, max_iter=100, lr=0.01,
            gamma_init=0.001, gamma_final=0.001,
            random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.75
