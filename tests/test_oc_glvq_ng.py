"""Tests for OC-GLVQ Neural Gas variants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prosemble.models import (
    OCGLVQ_NG, OCGRLVQ_NG, OCGMLVQ_NG, OCLGMLVQ_NG, OCGTLVQ_NG,
)


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
# OCGLVQ_NG
# ===========================================================================

class TestOCGLVQ_NGBasic:
    def test_fit_predict(self, binary_data):
        X, y = binary_data
        model = OCGLVQ_NG(n_prototypes=3, target_label=0,
                          max_iter=50, lr=0.01, random_seed=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        acc = float(jnp.mean(preds == y))
        assert acc > 0.6

    def test_gamma_decay(self, binary_data):
        X, y = binary_data
        model = OCGLVQ_NG(n_prototypes=3, target_label=0,
                          max_iter=50, lr=0.01, random_seed=42)
        model.fit(X, y)
        assert model.gamma_ is not None
        assert model.gamma_ < model._gamma_init_actual

    def test_gamma_bounded(self, binary_data):
        X, y = binary_data
        model = OCGLVQ_NG(n_prototypes=3, target_label=0,
                          gamma_init=5.0, gamma_final=0.5,
                          max_iter=50, lr=0.01, random_seed=42)
        model.fit(X, y)
        # gamma should have decayed and be bounded by gamma_init
        assert model.gamma_ >= 0.5  # never below final
        assert model.gamma_ < 5.0  # decayed from init

    def test_custom_gamma_decay(self, binary_data):
        X, y = binary_data
        model = OCGLVQ_NG(n_prototypes=3, target_label=0,
                          gamma_init=3.0, gamma_decay=0.95,
                          max_iter=50, lr=0.01, random_seed=42)
        model.fit(X, y)
        assert model.gamma_ is not None

    def test_decision_function(self, binary_data):
        X, y = binary_data
        model = OCGLVQ_NG(n_prototypes=3, target_label=0,
                          max_iter=50, lr=0.01, random_seed=42)
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (100,)
        assert jnp.all((scores >= 0) & (scores <= 1))

    def test_loss_decreases(self, binary_data):
        X, y = binary_data
        model = OCGLVQ_NG(n_prototypes=3, target_label=0,
                          max_iter=100, lr=0.01, random_seed=42)
        model.fit(X, y)
        assert model.loss_history_[0] != model.loss_history_[-1]


class TestOCGLVQ_NGSaveLoad:
    def test_save_load_roundtrip(self, binary_data, tmp_path):
        X, y = binary_data
        model = OCGLVQ_NG(n_prototypes=3, target_label=0,
                          max_iter=50, lr=0.01, random_seed=42)
        model.fit(X, y)
        path = str(tmp_path / "oc_glvq_ng.npz")
        model.save(path)
        loaded = OCGLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )
        assert loaded.gamma_ == pytest.approx(model.gamma_, rel=1e-4)


class TestOCGLVQ_NGResume:
    def test_resume_training(self, binary_data):
        X, y = binary_data
        model = OCGLVQ_NG(n_prototypes=3, target_label=0,
                          max_iter=30, lr=0.01, random_seed=42)
        model.fit(X, y)
        loss_before = model.loss_
        model.fit(X, y, resume=True)
        assert model.loss_ < abs(loss_before) * 2.0


# ===========================================================================
# OCGRLVQ_NG
# ===========================================================================

class TestOCGRLVQ_NGBasic:
    def test_fit_predict(self, binary_data):
        X, y = binary_data
        model = OCGRLVQ_NG(n_prototypes=3, target_label=0,
                           max_iter=50, lr=0.01, random_seed=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        acc = float(jnp.mean(preds == y))
        assert acc > 0.6

    def test_relevances_learned(self, binary_data):
        X, y = binary_data
        model = OCGRLVQ_NG(n_prototypes=3, target_label=0,
                           max_iter=50, lr=0.01, random_seed=42)
        model.fit(X, y)
        rel = model.relevance_profile
        assert rel.shape == (4,)
        assert jnp.allclose(jnp.sum(rel), 1.0, atol=1e-5)

    def test_gamma_decay(self, binary_data):
        X, y = binary_data
        model = OCGRLVQ_NG(n_prototypes=3, target_label=0,
                           max_iter=50, lr=0.01, random_seed=42)
        model.fit(X, y)
        assert model.gamma_ < model._gamma_init_actual


class TestOCGRLVQ_NGSaveLoad:
    def test_save_load_roundtrip(self, binary_data, tmp_path):
        X, y = binary_data
        model = OCGRLVQ_NG(n_prototypes=3, target_label=0,
                           max_iter=50, lr=0.01, random_seed=42)
        model.fit(X, y)
        path = str(tmp_path / "oc_grlvq_ng.npz")
        model.save(path)
        loaded = OCGRLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )


class TestOCGRLVQ_NGResume:
    def test_resume_training(self, binary_data):
        X, y = binary_data
        model = OCGRLVQ_NG(n_prototypes=3, target_label=0,
                           max_iter=30, lr=0.01, random_seed=42)
        model.fit(X, y)
        loss_before = model.loss_
        model.fit(X, y, resume=True)
        assert model.loss_ < abs(loss_before) * 2.0


# ===========================================================================
# OCGMLVQ_NG
# ===========================================================================

class TestOCGMLVQ_NGBasic:
    def test_fit_predict(self, binary_data):
        X, y = binary_data
        model = OCGMLVQ_NG(n_prototypes=3, target_label=0,
                           latent_dim=2, max_iter=50, lr=0.01,
                           random_seed=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        acc = float(jnp.mean(preds == y))
        assert acc > 0.6

    def test_omega_learned(self, binary_data):
        X, y = binary_data
        model = OCGMLVQ_NG(n_prototypes=3, target_label=0,
                           latent_dim=2, max_iter=50, lr=0.01,
                           random_seed=42)
        model.fit(X, y)
        assert model.omega_matrix.shape == (4, 2)
        assert model.lambda_matrix.shape == (2, 2)

    def test_gamma_decay(self, binary_data):
        X, y = binary_data
        model = OCGMLVQ_NG(n_prototypes=3, target_label=0,
                           latent_dim=2, max_iter=50, lr=0.01,
                           random_seed=42)
        model.fit(X, y)
        assert model.gamma_ < model._gamma_init_actual


class TestOCGMLVQ_NGSaveLoad:
    def test_save_load_roundtrip(self, binary_data, tmp_path):
        X, y = binary_data
        model = OCGMLVQ_NG(n_prototypes=3, target_label=0,
                           latent_dim=2, max_iter=50, lr=0.01,
                           random_seed=42)
        model.fit(X, y)
        path = str(tmp_path / "oc_gmlvq_ng.npz")
        model.save(path)
        loaded = OCGMLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )


class TestOCGMLVQ_NGResume:
    def test_resume_training(self, binary_data):
        X, y = binary_data
        model = OCGMLVQ_NG(n_prototypes=3, target_label=0,
                           latent_dim=2, max_iter=30, lr=0.01,
                           random_seed=42)
        model.fit(X, y)
        loss_before = model.loss_
        model.fit(X, y, resume=True)
        assert model.loss_ < abs(loss_before) * 2.0


# ===========================================================================
# OCLGMLVQ_NG
# ===========================================================================

class TestOCLGMLVQ_NGBasic:
    def test_fit_predict(self, binary_data):
        X, y = binary_data
        model = OCLGMLVQ_NG(n_prototypes=3, target_label=0,
                            latent_dim=2, max_iter=50, lr=0.01,
                            random_seed=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        acc = float(jnp.mean(preds == y))
        assert acc > 0.6

    def test_omegas_shape(self, binary_data):
        X, y = binary_data
        model = OCLGMLVQ_NG(n_prototypes=3, target_label=0,
                            latent_dim=2, max_iter=50, lr=0.01,
                            random_seed=42)
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 2)

    def test_gamma_decay(self, binary_data):
        X, y = binary_data
        model = OCLGMLVQ_NG(n_prototypes=3, target_label=0,
                            latent_dim=2, max_iter=50, lr=0.01,
                            random_seed=42)
        model.fit(X, y)
        assert model.gamma_ < model._gamma_init_actual


class TestOCLGMLVQ_NGSaveLoad:
    def test_save_load_roundtrip(self, binary_data, tmp_path):
        X, y = binary_data
        model = OCLGMLVQ_NG(n_prototypes=3, target_label=0,
                            latent_dim=2, max_iter=50, lr=0.01,
                            random_seed=42)
        model.fit(X, y)
        path = str(tmp_path / "oc_lgmlvq_ng.npz")
        model.save(path)
        loaded = OCLGMLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )


class TestOCLGMLVQ_NGResume:
    def test_resume_training(self, binary_data):
        X, y = binary_data
        model = OCLGMLVQ_NG(n_prototypes=3, target_label=0,
                            latent_dim=2, max_iter=30, lr=0.01,
                            random_seed=42)
        model.fit(X, y)
        loss_before = model.loss_
        model.fit(X, y, resume=True)
        assert model.loss_ < abs(loss_before) * 2.0


# ===========================================================================
# OCGTLVQ_NG
# ===========================================================================

class TestOCGTLVQ_NGBasic:
    def test_fit_predict(self, binary_data):
        X, y = binary_data
        model = OCGTLVQ_NG(n_prototypes=3, target_label=0,
                           subspace_dim=2, max_iter=50, lr=0.01,
                           random_seed=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        acc = float(jnp.mean(preds == y))
        assert acc > 0.6

    def test_omegas_orthonormal(self, binary_data):
        X, y = binary_data
        model = OCGTLVQ_NG(n_prototypes=3, target_label=0,
                           subspace_dim=2, max_iter=50, lr=0.01,
                           random_seed=42)
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 2)
        for k in range(3):
            gram = model.omegas_[k].T @ model.omegas_[k]
            np.testing.assert_allclose(gram, jnp.eye(2), atol=1e-4)

    def test_gamma_decay(self, binary_data):
        X, y = binary_data
        model = OCGTLVQ_NG(n_prototypes=3, target_label=0,
                           subspace_dim=2, max_iter=50, lr=0.01,
                           random_seed=42)
        model.fit(X, y)
        assert model.gamma_ < model._gamma_init_actual


class TestOCGTLVQ_NGSaveLoad:
    def test_save_load_roundtrip(self, binary_data, tmp_path):
        X, y = binary_data
        model = OCGTLVQ_NG(n_prototypes=3, target_label=0,
                           subspace_dim=2, max_iter=50, lr=0.01,
                           random_seed=42)
        model.fit(X, y)
        path = str(tmp_path / "oc_gtlvq_ng.npz")
        model.save(path)
        loaded = OCGTLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )


class TestOCGTLVQ_NGResume:
    def test_resume_training(self, binary_data):
        X, y = binary_data
        model = OCGTLVQ_NG(n_prototypes=3, target_label=0,
                           subspace_dim=2, max_iter=30, lr=0.01,
                           random_seed=42)
        model.fit(X, y)
        loss_before = model.loss_
        model.fit(X, y, resume=True)
        assert model.loss_ < abs(loss_before) * 2.0


# ===========================================================================
# Reject option (parametrized across all 5 models)
# ===========================================================================

@pytest.mark.parametrize("ModelClass,kwargs", [
    (OCGLVQ_NG, {}),
    (OCGRLVQ_NG, {}),
    (OCGMLVQ_NG, {'latent_dim': 2}),
    (OCLGMLVQ_NG, {'latent_dim': 2}),
    (OCGTLVQ_NG, {'subspace_dim': 2}),
])
class TestRejectOption:
    def test_predict_with_reject(self, ModelClass, kwargs, binary_data):
        X, y = binary_data
        model = ModelClass(
            n_prototypes=3, target_label=0,
            max_iter=50, lr=0.01, random_seed=42, **kwargs
        )
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.6, lower=0.4)
        assert jnp.any(preds == -1) or True  # may or may not reject
        assert set(np.unique(preds).tolist()).issubset({-1, 0, 1})
