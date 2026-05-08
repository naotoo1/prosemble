"""Tests for OC-RSLVQ Neural Gas variants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models import OCRSLVQ_NG, OCMRSLVQ_NG, OCLMRSLVQ_NG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Binary one-class dataset."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X_target = jax.random.normal(k1, (60, 4)) * 0.5
    X_outlier = jax.random.normal(k2, (40, 4)) * 0.5 + 3.0
    X = jnp.concatenate([X_target, X_outlier])
    y = jnp.concatenate([jnp.zeros(60, dtype=jnp.int32),
                         jnp.ones(40, dtype=jnp.int32)])
    return X, y


# ===========================================================================
# OCRSLVQ_NG
# ===========================================================================

class TestOCRSLVQ_NGBasic:
    def test_fit_predict(self, binary_data):
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        acc = float(jnp.mean(preds == y))
        assert acc > 0.6

    def test_gamma_decay(self, binary_data):
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.gamma_ is not None
        assert model.gamma_ < model._gamma_init_actual

    def test_gamma_bounded(self, binary_data):
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            gamma_init=5.0, gamma_final=0.5,
            max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.gamma_ >= 0.5
        assert model.gamma_ < 5.0

    def test_custom_gamma_decay(self, binary_data):
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            gamma_init=3.0, gamma_decay=0.95,
            max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.gamma_ is not None

    def test_decision_function(self, binary_data):
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (100,)
        assert jnp.all((scores >= 0) & (scores <= 1))

    def test_target_higher_scores(self, binary_data):
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            max_iter=100, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        mean_target = float(jnp.mean(scores[:60]))
        mean_outlier = float(jnp.mean(scores[60:]))
        assert mean_target > mean_outlier

    def test_loss_decreases(self, binary_data):
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            max_iter=100, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.loss_history_[0] != model.loss_history_[-1]

    def test_no_omega(self, binary_data):
        """OCRSLVQ_NG has no omega matrix — pure Euclidean."""
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            max_iter=30, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert not hasattr(model, 'omega_') or model.__dict__.get('omega_') is None
        assert not hasattr(model, 'omegas_') or model.__dict__.get('omegas_') is None

    def test_thetas_positive(self, binary_data):
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert jnp.all(model.thetas_ > 0)


class TestOCRSLVQ_NGSaveLoad:
    def test_save_load_roundtrip(self, binary_data, tmp_path):
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        path = str(tmp_path / "oc_rslvq_ng.npz")
        model.save(path)
        loaded = OCRSLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )
        assert loaded.gamma_ == pytest.approx(model.gamma_, rel=1e-4)


class TestOCRSLVQ_NGResume:
    def test_resume_training(self, binary_data):
        X, y = binary_data
        model = OCRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            max_iter=30, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        loss_before = model.loss_
        model.fit(X, y, resume=True)
        assert model.loss_ < abs(loss_before) * 2.0


# ===========================================================================
# OCMRSLVQ_NG
# ===========================================================================

class TestOCMRSLVQ_NGBasic:
    def test_fit_predict(self, binary_data):
        X, y = binary_data
        model = OCMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        acc = float(jnp.mean(preds == y))
        assert acc > 0.6

    def test_omega_learned(self, binary_data):
        X, y = binary_data
        model = OCMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.omega_matrix.shape == (4, 2)
        assert model.lambda_matrix.shape == (2, 2)

    def test_gamma_decay(self, binary_data):
        X, y = binary_data
        model = OCMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.gamma_ < model._gamma_init_actual

    def test_decision_function(self, binary_data):
        X, y = binary_data
        model = OCMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (100,)
        assert jnp.all((scores >= 0) & (scores <= 1))

    def test_thetas_positive(self, binary_data):
        X, y = binary_data
        model = OCMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert jnp.all(model.thetas_ > 0)


class TestOCMRSLVQ_NGSaveLoad:
    def test_save_load_roundtrip(self, binary_data, tmp_path):
        X, y = binary_data
        model = OCMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        path = str(tmp_path / "oc_mrslvq_ng.npz")
        model.save(path)
        loaded = OCMRSLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )


class TestOCMRSLVQ_NGResume:
    def test_resume_training(self, binary_data):
        X, y = binary_data
        model = OCMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=30, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        loss_before = model.loss_
        model.fit(X, y, resume=True)
        assert model.loss_ < abs(loss_before) * 2.0


# ===========================================================================
# OCLMRSLVQ_NG
# ===========================================================================

class TestOCLMRSLVQ_NGBasic:
    def test_fit_predict(self, binary_data):
        X, y = binary_data
        model = OCLMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        acc = float(jnp.mean(preds == y))
        assert acc > 0.6

    def test_omegas_shape(self, binary_data):
        X, y = binary_data
        model = OCLMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 2)

    def test_gamma_decay(self, binary_data):
        X, y = binary_data
        model = OCLMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.gamma_ < model._gamma_init_actual

    def test_decision_function(self, binary_data):
        X, y = binary_data
        model = OCLMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (100,)
        assert jnp.all((scores >= 0) & (scores <= 1))

    def test_thetas_positive(self, binary_data):
        X, y = binary_data
        model = OCLMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert jnp.all(model.thetas_ > 0)


class TestOCLMRSLVQ_NGSaveLoad:
    def test_save_load_roundtrip(self, binary_data, tmp_path):
        X, y = binary_data
        model = OCLMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=50, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        path = str(tmp_path / "oc_lmrslvq_ng.npz")
        model.save(path)
        loaded = OCLMRSLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )


class TestOCLMRSLVQ_NGResume:
    def test_resume_training(self, binary_data):
        X, y = binary_data
        model = OCLMRSLVQ_NG(
            sigma=0.5, n_prototypes=3, target_label=0,
            latent_dim=2, max_iter=30, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        loss_before = model.loss_
        model.fit(X, y, resume=True)
        assert model.loss_ < abs(loss_before) * 2.0


# ===========================================================================
# Reject option (parametrized across all 3 models)
# ===========================================================================

@pytest.mark.parametrize("ModelClass,kwargs", [
    (OCRSLVQ_NG, {'sigma': 0.5}),
    (OCMRSLVQ_NG, {'sigma': 0.5, 'latent_dim': 2}),
    (OCLMRSLVQ_NG, {'sigma': 0.5, 'latent_dim': 2}),
])
class TestRejectOption:
    def test_predict_with_reject(self, ModelClass, kwargs, binary_data):
        X, y = binary_data
        model = ModelClass(
            n_prototypes=3, target_label=0,
            max_iter=50, lr=0.01, random_seed=42, **kwargs,
        )
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.6, lower=0.4)
        assert set(np.unique(preds).tolist()).issubset({-1, 0, 1})
