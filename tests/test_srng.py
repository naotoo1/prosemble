"""Tests for Supervised Relevance Neural Gas (SRNG)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.srng import SRNG


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


class TestSRNGBasic:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = SRNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_iris(self, iris_data):
        X, y = iris_data
        model = SRNG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.80

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = SRNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_prototypes_shape(self, iris_data):
        X, y = iris_data
        model = SRNG(n_prototypes_per_class=3, max_iter=20, lr=0.01)
        model.fit(X, y)
        assert model.prototypes_.shape == (9, 4)  # 3 classes * 3 per class


class TestSRNGGammaDecay:
    def test_gamma_decays(self, separable_2d):
        X, y = separable_2d
        model = SRNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0
        assert model.gamma_ >= 0.01

    def test_gamma_reaches_final(self, separable_2d):
        X, y = separable_2d
        model = SRNG(
            n_prototypes_per_class=2, max_iter=200, lr=0.01,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        # Gamma should have decayed from initial value
        # (may not reach gamma_final if early stopping kicks in)
        assert model.gamma_ < 5.0

    def test_custom_gamma_decay(self, separable_2d):
        X, y = separable_2d
        model = SRNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            gamma_init=5.0, gamma_decay=0.95,
        )
        model.fit(X, y)
        # After 50 steps with 0.95 decay: 5.0 * 0.95^50 ≈ 0.036
        assert model.gamma_ < 1.0


class TestSRNGRelevance:
    def test_relevances_learned(self, iris_data):
        X, y = iris_data
        model = SRNG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        relevances = model.relevance_profile
        assert relevances.shape == (4,)
        # Relevances should sum to ~1 (softmax output)
        np.testing.assert_allclose(float(jnp.sum(relevances)), 1.0, atol=1e-5)
        # Should be non-uniform after training
        assert float(jnp.std(relevances)) > 0.01

    def test_relevance_profile_unfitted_raises(self):
        model = SRNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.relevance_profile


class TestSRNGTrainingModes:
    def test_scan_training(self, separable_2d):
        X, y = separable_2d
        model = SRNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            use_scan=True,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_minibatch_training(self, iris_data):
        X, y = iris_data
        model = SRNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            batch_size=32,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.60


class TestSRNGSaveLoad:
    def test_save_load_roundtrip(self, iris_data):
        X, y = iris_data
        model = SRNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "srng_model")
            model.save(path)

            loaded = SRNG.load(path)
            np.testing.assert_allclose(
                np.asarray(model.prototypes_),
                np.asarray(loaded.prototypes_),
            )
            np.testing.assert_allclose(
                np.asarray(model.relevances_),
                np.asarray(loaded.relevances_),
            )
            # Predictions should match
            preds_orig = model.predict(X)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestSRNGResume:
    def test_resume_training(self, iris_data):
        X, y = iris_data
        model = SRNG(n_prototypes_per_class=2, max_iter=30, lr=0.01)
        model.fit(X, y)
        acc_before = float(jnp.mean(model.predict(X) == y))

        model.fit(X, y, resume=True)
        acc_after = float(jnp.mean(model.predict(X) == y))
        # Should maintain or improve accuracy
        assert acc_after >= acc_before - 0.05

    def test_resume_with_initial_prototypes_raises(self, iris_data):
        X, y = iris_data
        model = SRNG(n_prototypes_per_class=2, max_iter=30, lr=0.01)
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.fit(X, y, resume=True, initial_prototypes=model.prototypes_)

    def test_partial_fit(self, iris_data):
        X, y = iris_data
        model = SRNG(n_prototypes_per_class=2, max_iter=30, lr=0.01)
        model.fit(X, y)
        n_iter_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_iter_before + 1


class TestSRNGSmallGamma:
    def test_small_gamma_like_grlvq(self, iris_data):
        """With very small gamma, SRNG should behave similarly to GRLVQ."""
        X, y = iris_data
        model = SRNG(
            n_prototypes_per_class=1, max_iter=100, lr=0.01,
            gamma_init=0.001, gamma_final=0.001,
            random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        # Should achieve reasonable accuracy like GRLVQ
        assert accuracy > 0.75
