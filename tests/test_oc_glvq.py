"""Tests for OC-GLVQ (One-Class GLVQ)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.oc_glvq import OCGLVQ


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
def iris_binary():
    """Iris binary OCC dataset."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X = jnp.array(data.data, dtype=jnp.float32)
    y = jnp.where(jnp.array(data.target) == 0, 0, 1).astype(jnp.int32)
    return X, y


# ============================================================
# Basic tests
# ============================================================

class TestOCGLVQBasic:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_iris_binary(self, iris_binary):
        X, y = iris_binary
        model = OCGLVQ(
            n_prototypes=3, max_iter=100, lr=0.01,
            target_label=0, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_mask = (y == 0)
        target_recall = float(jnp.mean(preds[target_mask] == 0))
        assert target_recall >= 0.60

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_prototypes_shape(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=5, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.prototypes_.shape == (5, 2)

    def test_auto_target_label(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=20, lr=0.01)
        model.fit(X, y)
        # Most frequent class (40 samples) should be auto-detected
        assert model._target_label == 0

    def test_prototype_labels_all_target(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        np.testing.assert_array_equal(model.prototype_labels_, 0)


# ============================================================
# Thetas tests
# ============================================================

class TestOCGLVQThetas:
    def test_thetas_shape(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=4, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.thetas_.shape == (4,)

    def test_thetas_positive(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert jnp.all(model.thetas_ > 0)

    def test_visibility_radii_property(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        np.testing.assert_array_equal(model.visibility_radii, model.thetas_)

    def test_visibility_radii_not_fitted(self):
        model = OCGLVQ(n_prototypes=3)
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.visibility_radii


# ============================================================
# Decision function tests
# ============================================================

class TestOCGLVQDecisionFunction:
    def test_target_higher_scores(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=100, lr=0.01, target_label=0)
        model.fit(X, y)
        scores = model.decision_function(X)
        mean_target = float(jnp.mean(scores[:40]))
        mean_outlier = float(jnp.mean(scores[40:]))
        assert mean_target > mean_outlier

    def test_scores_range(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        scores = model.decision_function(X)
        assert jnp.all(scores >= 0.0) and jnp.all(scores <= 1.0)

    def test_predict_proba_shape(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (60,)


# ============================================================
# Beta and margin tests
# ============================================================

class TestOCGLVQBetaMargin:
    def test_default_beta(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert np.isfinite(model.loss_)

    def test_custom_beta(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(
            n_prototypes=3, max_iter=50, lr=0.01,
            target_label=0, beta=5.0,
        )
        model.fit(X, y)
        assert np.isfinite(model.loss_)

    def test_positive_margin(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(
            n_prototypes=3, max_iter=50, lr=0.01,
            target_label=0, margin=0.1,
        )
        model.fit(X, y)
        assert np.isfinite(model.loss_)


# ============================================================
# Reject option tests
# ============================================================

class TestOCGLVQReject:
    def test_reject_option(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
        assert preds.shape == (60,)
        possible = {0, 1, -1}
        actual = set(int(p) for p in preds)
        assert actual.issubset(possible)

    def test_no_reject_zone(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict(X)
        preds_reject = model.predict_with_reject(X, upper=0.5, lower=0.5)
        np.testing.assert_array_equal(preds, preds_reject)


# ============================================================
# Training modes
# ============================================================

class TestOCGLVQTrainingModes:
    def test_scan_training(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(
            n_prototypes=3, max_iter=50, lr=0.01,
            target_label=0, use_scan=True,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.60

    def test_minibatch_training(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(
            n_prototypes=3, max_iter=50, lr=0.01,
            target_label=0, batch_size=16,
        )
        model.fit(X, y)
        assert model.thetas_ is not None


# ============================================================
# Save / load
# ============================================================

class TestOCGLVQSaveLoad:
    def test_save_load_roundtrip(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "oc_glvq_model")
            model.save(path)

            loaded = OCGLVQ.load(path)
            np.testing.assert_allclose(
                np.asarray(model.prototypes_),
                np.asarray(loaded.prototypes_),
            )
            np.testing.assert_allclose(
                np.asarray(model.thetas_),
                np.asarray(loaded.thetas_),
            )
            preds_orig = model.predict(X)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_orig, preds_loaded)


# ============================================================
# Resume / partial_fit
# ============================================================

class TestOCGLVQResume:
    def test_resume_training(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        loss_before = model.loss_
        model.fit(X, y, resume=True)
        # Loss can be negative (sigmoid output), so check it doesn't explode
        assert model.loss_ < abs(loss_before) * 2.0

    def test_partial_fit(self, occ_2d):
        X, y = occ_2d
        model = OCGLVQ(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        n_iter_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_iter_before + 1
