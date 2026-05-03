"""Tests for OC-GLVQ metric adaptation variants (R, M, LM, T)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.oc_grlvq import OCGRLVQ
from prosemble.models.oc_gmlvq import OCGMLVQ
from prosemble.models.oc_lgmlvq import OCLGMLVQ
from prosemble.models.oc_gtlvq import OCGTLVQ


@pytest.fixture
def occ_2d():
    """Simple 2D one-class dataset."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X_target = jax.random.normal(k1, (40, 2)) * 0.3
    X_outlier = jax.random.normal(k2, (20, 2)) * 0.3 + 3.0
    X = jnp.concatenate([X_target, X_outlier])
    y = jnp.concatenate([jnp.zeros(40, dtype=jnp.int32),
                         jnp.ones(20, dtype=jnp.int32)])
    return X, y


@pytest.fixture
def occ_4d():
    """4D dataset with 2 relevant and 2 irrelevant features."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n_target, n_outlier = 60, 30
    X_t_rel = jax.random.normal(k1, (n_target, 2)) * 0.3
    X_o_rel = jax.random.normal(k2, (n_outlier, 2)) * 0.3 + 3.0
    X_t_noise = jax.random.normal(k3, (n_target, 2)) * 2.0
    X_o_noise = jax.random.normal(k4, (n_outlier, 2)) * 2.0
    X = jnp.concatenate([
        jnp.concatenate([X_t_rel, X_t_noise], axis=1),
        jnp.concatenate([X_o_rel, X_o_noise], axis=1),
    ])
    y = jnp.concatenate([jnp.zeros(n_target, dtype=jnp.int32),
                         jnp.ones(n_outlier, dtype=jnp.int32)])
    return X, y


@pytest.fixture
def iris_binary():
    """Iris binary OCC dataset."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X = jnp.array(data.data, dtype=jnp.float32)
    y = jnp.where(jnp.array(data.target) == 0, 0, 1).astype(jnp.int32)
    return X, y


# ============================================================
# OC-GRLVQ Tests
# ============================================================

class TestOCGRLVQBasic:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = OCGRLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = OCGRLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_prototypes_shape(self, occ_2d):
        X, y = occ_2d
        model = OCGRLVQ(n_prototypes=5, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.prototypes_.shape == (5, 2)


class TestOCGRLVQRelevances:
    def test_relevances_shape(self, occ_2d):
        X, y = occ_2d
        model = OCGRLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.relevances_.shape == (2,)

    def test_relevances_sum_to_one(self, occ_2d):
        X, y = occ_2d
        model = OCGRLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        np.testing.assert_allclose(float(jnp.sum(model.relevances_)), 1.0, atol=1e-5)

    def test_relevances_positive(self, occ_2d):
        X, y = occ_2d
        model = OCGRLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert jnp.all(model.relevances_ > 0)

    def test_relevance_profile_property(self, occ_2d):
        X, y = occ_2d
        model = OCGRLVQ(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        np.testing.assert_array_equal(model.relevance_profile, model.relevances_)

    def test_relevance_profile_not_fitted(self):
        model = OCGRLVQ(n_prototypes=3)
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.relevance_profile

    def test_relevant_features_higher_weight(self, occ_4d):
        X, y = occ_4d
        model = OCGRLVQ(
            n_prototypes=3, max_iter=200, lr=0.01,
            target_label=0, random_seed=42,
        )
        model.fit(X, y)
        rel = model.relevances_
        relevant_weight = float(rel[0] + rel[1])
        noise_weight = float(rel[2] + rel[3])
        assert relevant_weight > noise_weight


class TestOCGRLVQSaveLoad:
    def test_save_load_roundtrip(self, occ_2d):
        X, y = occ_2d
        model = OCGRLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "oc_grlvq_model")
            model.save(path)
            loaded = OCGRLVQ.load(path)
            np.testing.assert_allclose(
                np.asarray(model.relevances_),
                np.asarray(loaded.relevances_),
            )
            np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


class TestOCGRLVQResume:
    def test_resume(self, occ_2d):
        X, y = occ_2d
        model = OCGRLVQ(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        model.fit(X, y, resume=True)
        assert model.relevances_ is not None

    def test_partial_fit(self, occ_2d):
        X, y = occ_2d
        model = OCGRLVQ(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        n_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_before + 1


# ============================================================
# OC-GMLVQ Tests
# ============================================================

class TestOCGMLVQBasic:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = OCGMLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_iris(self, iris_binary):
        X, y = iris_binary
        model = OCGMLVQ(
            n_prototypes=3, max_iter=100, lr=0.01,
            target_label=0, latent_dim=2, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_recall = float(jnp.mean(preds[y == 0] == 0))
        assert target_recall >= 0.60

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = OCGMLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestOCGMLVQOmega:
    def test_omega_shape_default(self, occ_2d):
        X, y = occ_2d
        model = OCGMLVQ(n_prototypes=3, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.omega_.shape == (2, 2)

    def test_omega_shape_custom(self, occ_2d):
        X, y = occ_2d
        model = OCGMLVQ(
            n_prototypes=3, max_iter=20, lr=0.01,
            target_label=0, latent_dim=1,
        )
        model.fit(X, y)
        assert model.omega_.shape == (2, 1)

    def test_omega_matrix_property(self, occ_2d):
        X, y = occ_2d
        model = OCGMLVQ(n_prototypes=3, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        np.testing.assert_array_equal(model.omega_matrix, model.omega_)

    def test_lambda_matrix(self, occ_2d):
        X, y = occ_2d
        model = OCGMLVQ(n_prototypes=3, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        lam = model.lambda_matrix
        assert lam.shape == (2, 2)
        # Λ = Ω^T Ω should be PSD — eigenvalues >= 0
        eigvals = jnp.linalg.eigvalsh(lam)
        assert jnp.all(eigvals >= -1e-6)


class TestOCGMLVQSaveLoad:
    def test_save_load(self, occ_2d):
        X, y = occ_2d
        model = OCGMLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "oc_gmlvq_model")
            model.save(path)
            loaded = OCGMLVQ.load(path)
            np.testing.assert_allclose(
                np.asarray(model.omega_), np.asarray(loaded.omega_),
            )
            np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


class TestOCGMLVQResume:
    def test_resume(self, occ_2d):
        X, y = occ_2d
        model = OCGMLVQ(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        model.fit(X, y, resume=True)
        assert model.omega_ is not None

    def test_partial_fit(self, occ_2d):
        X, y = occ_2d
        model = OCGMLVQ(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        n_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_before + 1


# ============================================================
# OC-LGMLVQ Tests
# ============================================================

class TestOCLGMLVQBasic:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = OCLGMLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_iris(self, iris_binary):
        X, y = iris_binary
        model = OCLGMLVQ(
            n_prototypes=3, max_iter=100, lr=0.01,
            target_label=0, latent_dim=2, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_recall = float(jnp.mean(preds[y == 0] == 0))
        assert target_recall >= 0.60

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = OCLGMLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestOCLGMLVQOmegas:
    def test_omegas_shape_default(self, occ_2d):
        X, y = occ_2d
        model = OCLGMLVQ(n_prototypes=3, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.omegas_.shape == (3, 2, 2)

    def test_omegas_shape_custom(self, occ_2d):
        X, y = occ_2d
        model = OCLGMLVQ(
            n_prototypes=3, max_iter=20, lr=0.01,
            target_label=0, latent_dim=1,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 2, 1)


class TestOCLGMLVQSaveLoad:
    def test_save_load(self, occ_2d):
        X, y = occ_2d
        model = OCLGMLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "oc_lgmlvq_model")
            model.save(path)
            loaded = OCLGMLVQ.load(path)
            np.testing.assert_allclose(
                np.asarray(model.omegas_), np.asarray(loaded.omegas_),
            )
            np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


class TestOCLGMLVQResume:
    def test_resume(self, occ_2d):
        X, y = occ_2d
        model = OCLGMLVQ(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        model.fit(X, y, resume=True)
        assert model.omegas_ is not None

    def test_partial_fit(self, occ_2d):
        X, y = occ_2d
        model = OCLGMLVQ(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        n_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_before + 1


# ============================================================
# OC-GTLVQ Tests
# ============================================================

class TestOCGTLVQBasic:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = OCGTLVQ(
            n_prototypes=3, max_iter=50, lr=0.01,
            target_label=0, subspace_dim=1,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.60

    def test_iris(self, iris_binary):
        X, y = iris_binary
        model = OCGTLVQ(
            n_prototypes=3, max_iter=100, lr=0.01,
            target_label=0, subspace_dim=2, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_recall = float(jnp.mean(preds[y == 0] == 0))
        assert target_recall >= 0.60

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = OCGTLVQ(
            n_prototypes=3, max_iter=50, lr=0.01,
            target_label=0, subspace_dim=1,
        )
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]


class TestOCGTLVQOmegas:
    def test_omegas_shape(self, occ_2d):
        X, y = occ_2d
        model = OCGTLVQ(
            n_prototypes=3, max_iter=20, lr=0.01,
            target_label=0, subspace_dim=1,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 2, 1)

    def test_omegas_orthonormal(self, occ_2d):
        X, y = occ_2d
        model = OCGTLVQ(
            n_prototypes=3, max_iter=50, lr=0.01,
            target_label=0, subspace_dim=1,
        )
        model.fit(X, y)
        for k in range(3):
            gram = model.omegas_[k].T @ model.omegas_[k]
            np.testing.assert_allclose(
                np.asarray(gram), np.eye(1), atol=1e-4,
            )

    def test_custom_subspace_dim(self, iris_binary):
        X, y = iris_binary
        model = OCGTLVQ(
            n_prototypes=3, max_iter=20, lr=0.01,
            target_label=0, subspace_dim=3,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 3)


class TestOCGTLVQSaveLoad:
    def test_save_load(self, occ_2d):
        X, y = occ_2d
        model = OCGTLVQ(
            n_prototypes=3, max_iter=50, lr=0.01,
            target_label=0, subspace_dim=1,
        )
        model.fit(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "oc_gtlvq_model")
            model.save(path)
            loaded = OCGTLVQ.load(path)
            np.testing.assert_allclose(
                np.asarray(model.omegas_), np.asarray(loaded.omegas_),
            )
            np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


class TestOCGTLVQResume:
    def test_resume(self, occ_2d):
        X, y = occ_2d
        model = OCGTLVQ(
            n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0, subspace_dim=1,
        )
        model.fit(X, y)
        model.fit(X, y, resume=True)
        assert model.omegas_ is not None

    def test_partial_fit(self, occ_2d):
        X, y = occ_2d
        model = OCGTLVQ(
            n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0, subspace_dim=1,
        )
        model.fit(X, y)
        n_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_before + 1


# ============================================================
# Reject option (inherited) — spot check for all variants
# ============================================================

class TestOCGLVQVariantsReject:
    @pytest.mark.parametrize("ModelClass,kwargs", [
        (OCGRLVQ, {}),
        (OCGMLVQ, {'latent_dim': 2}),
        (OCLGMLVQ, {'latent_dim': 2}),
        (OCGTLVQ, {'subspace_dim': 1}),
    ])
    def test_reject_inherited(self, occ_2d, ModelClass, kwargs):
        X, y = occ_2d
        model = ModelClass(
            n_prototypes=3, max_iter=50, lr=0.01,
            target_label=0, **kwargs,
        )
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
        assert preds.shape == (60,)
