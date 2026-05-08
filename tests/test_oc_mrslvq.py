"""Tests for OC-MRSLVQ and OC-LMRSLVQ."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.oc_mrslvq import OCMRSLVQ, OCLMRSLVQ


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


class TestOCMRSLVQ:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = OCMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = OCMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_thetas_positive(self, occ_2d):
        X, y = occ_2d
        model = OCMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        assert jnp.all(model.thetas_ > 0)

    def test_omega_shape(self, occ_2d):
        X, y = occ_2d
        model = OCMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=30, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        assert model.omega_matrix.shape == (2, 2)

    def test_lambda_matrix(self, occ_2d):
        X, y = occ_2d
        model = OCMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=30, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        lam = model.lambda_matrix
        assert lam.shape == (2, 2)
        np.testing.assert_allclose(lam, lam.T, atol=1e-5)

    def test_latent_dim(self, iris_binary):
        X, y = iris_binary
        model = OCMRSLVQ(
            sigma=1.0, latent_dim=2, n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0,
        )
        model.fit(X, y)
        assert model.omega_matrix.shape == (4, 2)

    def test_target_higher_scores(self, occ_2d):
        X, y = occ_2d
        model = OCMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=100, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        mean_target = float(jnp.mean(scores[:40]))
        mean_outlier = float(jnp.mean(scores[40:]))
        assert mean_target > mean_outlier

    def test_scores_range(self, occ_2d):
        X, y = occ_2d
        model = OCMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        assert jnp.all(scores >= 0.0) and jnp.all(scores <= 1.0)

    def test_reject_option(self, occ_2d):
        X, y = occ_2d
        model = OCMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
        assert preds.shape == (60,)
        possible = {0, 1, -1}
        actual = set(int(p) for p in preds)
        assert actual.issubset(possible)

    def test_save_load(self, occ_2d):
        X, y = occ_2d
        model = OCMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "oc_mrslvq")
            model.save(path)
            loaded = OCMRSLVQ.load(path)
            preds_after = loaded.predict(X)
            np.testing.assert_array_equal(preds_before, preds_after)


class TestOCLMRSLVQ:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = OCLMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = OCLMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_thetas_positive(self, occ_2d):
        X, y = occ_2d
        model = OCLMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        assert jnp.all(model.thetas_ > 0)

    def test_local_omegas_shape(self, occ_2d):
        X, y = occ_2d
        model = OCLMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=30, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 2, 2)  # 3 protos, 2x2 omegas

    def test_latent_dim(self, iris_binary):
        X, y = iris_binary
        model = OCLMRSLVQ(
            sigma=1.0, latent_dim=2, n_prototypes=3, max_iter=30, lr=0.01,
            target_label=0,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 2)

    def test_target_higher_scores(self, occ_2d):
        X, y = occ_2d
        model = OCLMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=100, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        mean_target = float(jnp.mean(scores[:40]))
        mean_outlier = float(jnp.mean(scores[40:]))
        assert mean_target > mean_outlier

    def test_scores_range(self, occ_2d):
        X, y = occ_2d
        model = OCLMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        assert jnp.all(scores >= 0.0) and jnp.all(scores <= 1.0)

    def test_save_load(self, occ_2d):
        X, y = occ_2d
        model = OCLMRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "oc_lmrslvq")
            model.save(path)
            loaded = OCLMRSLVQ.load(path)
            preds_after = loaded.predict(X)
            np.testing.assert_array_equal(preds_before, preds_after)
