"""Tests for Supervised Class-wise Matrix Neural Gas (SCMNG)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os

from prosemble.models.scmng import SCMNG


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


class TestSCMNGBasic:
    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = SCMNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_iris(self, iris_data):
        X, y = iris_data
        model = SCMNG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.80

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = SCMNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        assert model.loss_history_[-1] < model.loss_history_[0]

    def test_prototypes_shape(self, iris_data):
        X, y = iris_data
        model = SCMNG(n_prototypes_per_class=3, max_iter=20, lr=0.01)
        model.fit(X, y)
        assert model.prototypes_.shape == (9, 4)


class TestSCMNGGammaDecay:
    def test_gamma_decays(self, separable_2d):
        X, y = separable_2d
        model = SCMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0
        assert model.gamma_ >= 0.01

    def test_gamma_reaches_final(self, separable_2d):
        X, y = separable_2d
        model = SCMNG(
            n_prototypes_per_class=2, max_iter=200, lr=0.01,
            gamma_init=5.0, gamma_final=0.01,
        )
        model.fit(X, y)
        assert model.gamma_ < 5.0

    def test_custom_gamma_decay(self, separable_2d):
        X, y = separable_2d
        model = SCMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            gamma_init=5.0, gamma_decay=0.95,
        )
        model.fit(X, y)
        assert model.gamma_ < 1.0


class TestSCMNGOmegas:
    def test_omegas_shape(self, iris_data):
        X, y = iris_data
        model = SCMNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        # 3 classes, each with (4, 4) omega (no latent_dim reduction)
        assert model.omegas_.shape == (3, 4, 4)

    def test_omegas_latent_dim(self, iris_data):
        X, y = iris_data
        model = SCMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            latent_dim=2,
        )
        model.fit(X, y)
        assert model.omegas_.shape == (3, 4, 2)

    def test_omegas_differ_across_classes(self, iris_data):
        X, y = iris_data
        model = SCMNG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        # After training, class-wise omegas should diverge from each other
        omega_0 = model.omegas_[0]
        omega_1 = model.omegas_[1]
        diff = float(jnp.sum((omega_0 - omega_1) ** 2))
        assert diff > 1e-4

    def test_per_class_not_per_prototype(self, iris_data):
        X, y = iris_data
        model = SCMNG(n_prototypes_per_class=3, max_iter=20, lr=0.01)
        model.fit(X, y)
        # 3 classes -> 3 omegas, NOT 9 (one per prototype)
        assert model.omegas_.shape[0] == 3


class TestSCMNGLrRatio:
    def test_lr_ratio_default(self, separable_2d):
        X, y = separable_2d
        model = SCMNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        assert model.lr_ratio == 1.0

    def test_lr_ratio_trains(self, iris_data):
        X, y = iris_data
        model = SCMNG(
            n_prototypes_per_class=2, max_iter=100, lr=0.01,
            lr_ratio=0.5,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.70

    def test_lr_ratio_changes_result(self, iris_data):
        X, y = iris_data
        model_1 = SCMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            lr_ratio=1.0, random_seed=42,
        )
        model_1.fit(X, y)

        model_05 = SCMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            lr_ratio=0.5, random_seed=42,
        )
        model_05.fit(X, y)

        # Different lr_ratio should produce different prototypes
        diff = float(jnp.sum((model_1.prototypes_ - model_05.prototypes_) ** 2))
        assert diff > 1e-6


class TestSCMNGOmegaLr:
    def test_omega_lr_trains(self, iris_data):
        X, y = iris_data
        model = SCMNG(
            n_prototypes_per_class=2, max_iter=100, lr=0.01,
            omega_lr=0.001,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.70


class TestSCMNGTrainingModes:
    def test_scan_training(self, separable_2d):
        X, y = separable_2d
        model = SCMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            use_scan=True,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.75

    def test_minibatch_training(self, iris_data):
        X, y = iris_data
        model = SCMNG(
            n_prototypes_per_class=2, max_iter=50, lr=0.01,
            batch_size=32,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy >= 0.60


class TestSCMNGSaveLoad:
    def test_save_load_roundtrip(self, iris_data):
        X, y = iris_data
        model = SCMNG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scmng_model")
            model.save(path)

            loaded = SCMNG.load(path)
            np.testing.assert_allclose(
                np.asarray(model.prototypes_),
                np.asarray(loaded.prototypes_),
            )
            np.testing.assert_allclose(
                np.asarray(model.omegas_),
                np.asarray(loaded.omegas_),
            )
            preds_orig = model.predict(X)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestSCMNGResume:
    def test_resume_training(self, iris_data):
        X, y = iris_data
        model = SCMNG(n_prototypes_per_class=2, max_iter=30, lr=0.01)
        model.fit(X, y)
        acc_before = float(jnp.mean(model.predict(X) == y))

        model.fit(X, y, resume=True)
        acc_after = float(jnp.mean(model.predict(X) == y))
        assert acc_after >= acc_before - 0.05

    def test_partial_fit(self, iris_data):
        X, y = iris_data
        model = SCMNG(n_prototypes_per_class=2, max_iter=30, lr=0.01)
        model.fit(X, y)
        n_iter_before = model.n_iter_
        model.partial_fit(X, y)
        assert model.n_iter_ == n_iter_before + 1


class TestSCMNGSmallGamma:
    def test_small_gamma_like_gmlvq(self, iris_data):
        """With very small gamma, SCMNG should behave similarly to GMLVQ."""
        X, y = iris_data
        model = SCMNG(
            n_prototypes_per_class=1, max_iter=100, lr=0.01,
            gamma_init=0.001, gamma_final=0.001,
            random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.75
