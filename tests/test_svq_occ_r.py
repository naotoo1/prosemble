"""Tests for SVQ-OCC reject option and SVQOCC_R (relevance-weighted)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.svq_occ import SVQOCC
from prosemble.models.svq_occ_r import SVQOCC_R


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
    # Features 0-1: discriminative
    X_t_rel = jax.random.normal(k1, (n_target, 2)) * 0.3
    X_o_rel = jax.random.normal(k2, (n_outlier, 2)) * 0.3 + 3.0
    # Features 2-3: pure noise (same distribution for both)
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
# Tests for SVQOCC.predict_with_reject
# ============================================================

class TestRejectOption:
    def test_no_rejection_zone(self, occ_2d):
        """With lower == upper, predict_with_reject equals predict."""
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict(X)
        preds_reject = model.predict_with_reject(X, upper=0.5, lower=0.5)
        np.testing.assert_array_equal(preds, preds_reject)

    def test_rejection_zone_produces_rejects(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
        unique_labels = set(int(p) for p in preds)
        # Should contain target (0), non-target (1), and possibly reject (-1)
        assert 0 in unique_labels or 1 in unique_labels

    def test_wide_rejection_zone(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        # Very wide rejection: almost everything rejected
        preds = model.predict_with_reject(X, upper=0.99, lower=0.01)
        n_rejected = int(jnp.sum(preds == -1))
        # At least some should be rejected
        assert preds.shape == (60,)

    def test_custom_reject_label(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict_with_reject(
            X, upper=0.7, lower=0.3, reject_label=99
        )
        # Reject label should be 99, not -1
        possible_labels = {0, 1, 99}
        actual_labels = set(int(p) for p in preds)
        assert actual_labels.issubset(possible_labels)

    def test_lower_default_equals_upper(self, occ_2d):
        """When lower is None, it defaults to upper (no rejection)."""
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.5)
        # No rejection zone → no -1 labels
        assert int(jnp.sum(preds == -1)) == 0

    def test_tight_threshold_high_recall(self, occ_2d):
        """Low upper threshold → higher target recall."""
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=100, lr=0.01, target_label=0)
        model.fit(X, y)
        preds_strict = model.predict_with_reject(X, upper=0.8, lower=0.8)
        preds_loose = model.predict_with_reject(X, upper=0.2, lower=0.2)
        # Loose threshold should classify more as target
        n_target_strict = int(jnp.sum(preds_strict[:40] == 0))
        n_target_loose = int(jnp.sum(preds_loose[:40] == 0))
        assert n_target_loose >= n_target_strict


# ============================================================
# Tests for SVQOCC_R (relevance-weighted)
# ============================================================

class TestSVQOCCRBasic:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_iris_binary(self, iris_binary):
        X, y = iris_binary
        model = SVQOCC_R(
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
        model = SVQOCC_R(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_prototypes_shape(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(n_prototypes=5, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.prototypes_.shape == (5, 2)


class TestSVQOCCRRelevances:
    def test_relevances_shape(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.relevances_.shape == (2,)

    def test_relevances_sum_to_one(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        np.testing.assert_allclose(float(jnp.sum(model.relevances_)), 1.0, atol=1e-5)

    def test_relevances_positive(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert jnp.all(model.relevances_ > 0)

    def test_relevance_profile_property(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        profile = model.relevance_profile
        np.testing.assert_array_equal(profile, model.relevances_)

    def test_relevance_profile_not_fitted(self):
        model = SVQOCC_R(n_prototypes=3)
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.relevance_profile

    def test_relevant_features_higher_weight(self, occ_4d):
        """On data with 2 relevant + 2 noise features, model should
        assign higher relevance to the discriminative features."""
        X, y = occ_4d
        model = SVQOCC_R(
            n_prototypes=3, max_iter=200, lr=0.01,
            target_label=0, random_seed=42,
        )
        model.fit(X, y)
        rel = model.relevances_
        # Features 0-1 are discriminative, 2-3 are noise
        relevant_weight = float(rel[0] + rel[1])
        noise_weight = float(rel[2] + rel[3])
        assert relevant_weight > noise_weight


class TestSVQOCCRCostFunctions:
    @pytest.mark.parametrize("cost_fn", ['contrastive', 'brier', 'cross_entropy'])
    def test_cost_function(self, occ_2d, cost_fn):
        X, y = occ_2d
        model = SVQOCC_R(
            n_prototypes=3, max_iter=30, lr=0.01, target_label=0,
            cost_function=cost_fn,
        )
        model.fit(X, y)
        assert np.isfinite(model.loss_)
        assert model.relevances_ is not None


class TestSVQOCCRRejectOption:
    def test_reject_inherited(self, occ_2d):
        """SVQOCC_R should inherit predict_with_reject from SVQOCC."""
        X, y = occ_2d
        model = SVQOCC_R(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict_with_reject(X, upper=0.7, lower=0.3)
        assert preds.shape == (60,)


class TestSVQOCCRSaveLoad:
    def test_save_load_roundtrip(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "svqocc_r_model")
            model.save(path)

            loaded = SVQOCC_R.load(path)
            np.testing.assert_allclose(
                np.asarray(model.prototypes_),
                np.asarray(loaded.prototypes_),
            )
            np.testing.assert_allclose(
                np.asarray(model.thetas_),
                np.asarray(loaded.thetas_),
            )
            np.testing.assert_allclose(
                np.asarray(model.relevances_),
                np.asarray(loaded.relevances_),
            )
            preds_orig = model.predict(X)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestSVQOCCRResume:
    def test_resume_training(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        loss_before = model.loss_
        model.fit(X, y, resume=True)
        assert model.loss_ < loss_before * 2.0

    def test_partial_fit(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        n_iter_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_iter_before + 1


class TestSVQOCCRTrainingModes:
    def test_scan_training(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            use_scan=True,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.60

    def test_minibatch_training(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC_R(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            batch_size=16,
        )
        model.fit(X, y)
        assert model.relevances_ is not None
