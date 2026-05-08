"""Tests for OC-RSLVQ."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.oc_rslvq import OCRSLVQ


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


class TestOCRSLVQ:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = OCRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = OCRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_thetas_positive(self, occ_2d):
        X, y = occ_2d
        model = OCRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        assert jnp.all(model.thetas_ > 0)

    def test_no_omega(self, occ_2d):
        """OC-RSLVQ has no omega matrix — pure Euclidean."""
        X, y = occ_2d
        model = OCRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=30, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        assert not hasattr(model, 'omega_') or model.__dict__.get('omega_') is None
        assert not hasattr(model, 'omegas_') or model.__dict__.get('omegas_') is None

    def test_target_higher_scores(self, occ_2d):
        X, y = occ_2d
        model = OCRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=100, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        mean_target = float(jnp.mean(scores[:40]))
        mean_outlier = float(jnp.mean(scores[40:]))
        assert mean_target > mean_outlier

    def test_scores_range(self, occ_2d):
        X, y = occ_2d
        model = OCRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        assert jnp.all(scores >= 0.0) and jnp.all(scores <= 1.0)

    def test_reject_option(self, occ_2d):
        X, y = occ_2d
        model = OCRSLVQ(
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
        model = OCRSLVQ(
            sigma=0.5, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "oc_rslvq")
            model.save(path)
            loaded = OCRSLVQ.load(path)
            preds_after = loaded.predict(X)
            np.testing.assert_array_equal(preds_before, preds_after)

    def test_iris(self, iris_binary):
        X, y = iris_binary
        model = OCRSLVQ(
            sigma=1.0, n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (150,)
        assert jnp.all(scores >= 0.0) and jnp.all(scores <= 1.0)
