"""Tests for Supervised Vector Quantization One-Class Classification (SVQ-OCC)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.svq_occ import SVQOCC


@pytest.fixture
def occ_2d():
    """Simple 2D one-class dataset: cluster at origin (target) vs distant points."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    # Target class: tight cluster around (0, 0)
    X_target = jax.random.normal(k1, (40, 2)) * 0.3
    # Non-target: points far from origin
    X_outlier = jax.random.normal(k2, (20, 2)) * 0.3 + 3.0
    X = jnp.concatenate([X_target, X_outlier])
    y = jnp.concatenate([jnp.zeros(40, dtype=jnp.int32),
                         jnp.ones(20, dtype=jnp.int32)])
    return X, y


@pytest.fixture
def iris_binary():
    """Iris dataset converted to binary OCC (class 0 = target, rest = outlier)."""
    from sklearn.datasets import load_iris
    data = load_iris()
    X = jnp.array(data.data, dtype=jnp.float32)
    y_orig = jnp.array(data.target, dtype=jnp.int32)
    # Class 0 is target (label 0), classes 1 and 2 are non-target (label 1)
    y = jnp.where(y_orig == 0, 0, 1)
    return X, y


class TestSVQOCCBasic:
    def test_fit_predict(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        preds = model.predict(X)
        # Should correctly classify most target samples
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.70

    def test_iris_binary(self, iris_binary):
        X, y = iris_binary
        model = SVQOCC(
            n_prototypes=3, max_iter=100, lr=0.01,
            target_label=0, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        # Target class recall
        target_mask = (y == 0)
        target_recall = float(jnp.mean(preds[target_mask] == 0))
        assert target_recall >= 0.60

    def test_loss_decreases(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_prototypes_shape(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(n_prototypes=5, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.prototypes_.shape == (5, 2)

    def test_auto_target_label(self, occ_2d):
        X, y = occ_2d
        # Class 0 has 40 samples (most frequent), should be auto-detected
        model = SVQOCC(n_prototypes=3, max_iter=20, lr=0.01)
        model.fit(X, y)
        assert model._target_label == 0

    def test_prototype_labels_all_target(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=20, lr=0.01, target_label=0)
        model.fit(X, y)
        # All prototype labels should be the target label
        assert jnp.all(model.prototype_labels_ == 0)


class TestSVQOCCCostFunctions:
    def test_contrastive(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01,
            cost_function='contrastive', target_label=0,
        )
        model.fit(X, y)
        assert model.loss_ is not None
        assert np.isfinite(model.loss_)

    def test_brier(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01,
            cost_function='brier', target_label=0,
        )
        model.fit(X, y)
        assert model.loss_ is not None
        assert np.isfinite(model.loss_)

    def test_cross_entropy(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01,
            cost_function='cross_entropy', target_label=0,
        )
        model.fit(X, y)
        assert model.loss_ is not None
        assert np.isfinite(model.loss_)

    def test_invalid_cost_function(self):
        with pytest.raises(ValueError, match="cost_function"):
            SVQOCC(cost_function='invalid')


class TestSVQOCCResponseTypes:
    def test_gaussian(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01,
            response_type='gaussian', target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (60,)
        assert jnp.all(scores >= 0) and jnp.all(scores <= 1)

    def test_student_t(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01,
            response_type='student_t', target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (60,)

    def test_uniform(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01,
            response_type='uniform', target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (60,)

    def test_invalid_response_type(self):
        with pytest.raises(ValueError, match="response_type"):
            SVQOCC(response_type='invalid')


class TestSVQOCCThetas:
    def test_thetas_shape(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(n_prototypes=4, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        assert model.thetas_.shape == (4,)

    def test_thetas_positive(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=50, lr=0.01, target_label=0)
        model.fit(X, y)
        assert jnp.all(model.thetas_ > 0)

    def test_visibility_radii_property(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        radii = model.visibility_radii
        np.testing.assert_array_equal(radii, model.thetas_)

    def test_visibility_radii_not_fitted(self):
        model = SVQOCC(n_prototypes=3)
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.visibility_radii


class TestSVQOCCLambdaDecay:
    def test_lambda_decays(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            lambda_init=5.0, lambda_final=0.01,
        )
        model.fit(X, y)
        assert model.lambda_ < 5.0
        assert model.lambda_ >= 0.01

    def test_custom_lambda_decay(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            lambda_init=5.0, lambda_decay=0.95,
        )
        model.fit(X, y)
        assert model.lambda_ < 5.0


class TestSVQOCCDecisionFunction:
    def test_target_higher_scores(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=100, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        scores = model.decision_function(X)
        # Target samples should generally have higher scores
        mean_target = float(jnp.mean(scores[:40]))
        mean_outlier = float(jnp.mean(scores[40:]))
        assert mean_target > mean_outlier

    def test_predict_proba_shape(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(n_prototypes=3, max_iter=30, lr=0.01, target_label=0)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (60,)
        assert jnp.all(proba >= 0) and jnp.all(proba <= 1)


class TestSVQOCCTrainingModes:
    def test_scan_training(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            use_scan=True,
        )
        model.fit(X, y)
        preds = model.predict(X)
        target_acc = float(jnp.mean(preds[:40] == 0))
        assert target_acc >= 0.60

    def test_minibatch_training(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            batch_size=16,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)


class TestSVQOCCSaveLoad:
    def test_save_load_roundtrip(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
        )
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "svqocc_model")
            model.save(path)

            loaded = SVQOCC.load(path)
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


class TestSVQOCCResume:
    def test_resume_training(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=30, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        loss_before = model.loss_

        model.fit(X, y, resume=True)
        # Loss should not increase dramatically
        assert model.loss_ < loss_before * 2.0

    def test_partial_fit(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=30, lr=0.01, target_label=0,
        )
        model.fit(X, y)
        n_iter_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_iter_before + 1


class TestSVQOCCAlpha:
    def test_alpha_zero_classification_only(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            alpha=0.0,  # Only classification cost
        )
        model.fit(X, y)
        assert model.loss_ is not None

    def test_alpha_one_representation_only(self, occ_2d):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=50, lr=0.01, target_label=0,
            alpha=1.0,  # Only representation cost
        )
        model.fit(X, y)
        assert model.loss_ is not None


class TestSVQOCCAllCombinations:
    """Test all 9 combinations of cost_function × response_type."""

    @pytest.mark.parametrize("cost_fn", ['contrastive', 'brier', 'cross_entropy'])
    @pytest.mark.parametrize("resp_type", ['gaussian', 'student_t', 'uniform'])
    def test_combination(self, occ_2d, cost_fn, resp_type):
        X, y = occ_2d
        model = SVQOCC(
            n_prototypes=3, max_iter=30, lr=0.01, target_label=0,
            cost_function=cost_fn, response_type=resp_type,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)
        assert np.isfinite(model.loss_)
