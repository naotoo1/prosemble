"""Tests for supervised Differentiating Kernel models with Neural Gas cooperation."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_iris

from prosemble.models import DKGLVQ_NG, DKGRLVQ_NG, DKGMLVQ_NG


@pytest.fixture
def separable_2d():
    """Simple 2D separable dataset (2 classes, 4 per class)."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X0 = jax.random.normal(k1, (8, 2)) * 0.3 - 1.5
    X1 = jax.random.normal(k2, (8, 2)) * 0.3 + 1.5
    X = jnp.concatenate([X0, X1])
    y = jnp.concatenate([jnp.zeros(8, dtype=jnp.int32),
                         jnp.ones(8, dtype=jnp.int32)])
    return X, y


@pytest.fixture
def iris_data():
    """Iris dataset (3 classes, 4 features)."""
    iris = load_iris()
    X = jnp.asarray(iris.data, dtype=jnp.float32)
    y = jnp.asarray(iris.target, dtype=jnp.int32)
    return X, y


@pytest.fixture
def separable_4d():
    """4D dataset with 2 relevant + 2 noise features."""
    key = jax.random.PRNGKey(123)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    X0_rel = jax.random.normal(k1, (15, 2)) * 0.3 - 1.5
    X1_rel = jax.random.normal(k2, (15, 2)) * 0.3 + 1.5
    X0_noise = jax.random.normal(k3, (15, 2)) * 2.0
    X1_noise = jax.random.normal(k4, (15, 2)) * 2.0
    X0 = jnp.concatenate([X0_rel, X0_noise], axis=1)
    X1 = jnp.concatenate([X1_rel, X1_noise], axis=1)
    X = jnp.concatenate([X0, X1])
    y = jnp.concatenate([jnp.zeros(15, dtype=jnp.int32),
                         jnp.ones(15, dtype=jnp.int32)])
    return X, y


# ============================================================================
# DKGLVQ_NG Tests
# ============================================================================

class TestDKGLVQ_NG:

    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                          use_scan=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (16,)
        acc = float(jnp.mean(preds == y))
        assert acc >= 0.75

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=100, lr=0.01,
                          use_scan=False)
        model.fit(X, y)
        history = model.loss_history_
        assert float(history[-1]) < float(history[0])

    def test_iris_accuracy(self, iris_data):
        X, y = iris_data
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        acc = float(jnp.mean(preds == y))
        assert acc >= 0.80

    def test_gamma_decays(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        assert model.gamma_ is not None
        assert model.gamma_ < model._gamma_init_actual

    def test_gamma_bounded(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                          gamma_init=5.0, gamma_final=0.5)
        model.fit(X, y)
        assert model.gamma_ >= 0.5
        assert model.gamma_ < 5.0

    def test_sigmas_learned(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                          use_scan=False)
        model.fit(X, y)
        assert model.sigmas_ is not None
        assert model.sigmas_.shape == (4,)  # 2 classes x 2 prototypes
        assert jnp.all(model.sigmas_ > 0)

    def test_sigma_init_fixed(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=30, lr=0.01,
                          sigma_init=1.0)
        model.fit(X, y)
        assert model.sigmas_ is not None

    def test_kernel_bandwidths_property(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=20, lr=0.01)
        model.fit(X, y)
        bw = model.kernel_bandwidths
        assert bw.shape == (4,)
        assert jnp.all(bw >= model.sigma_min)

    def test_python_loop(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=30, lr=0.01,
                          use_scan=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (16,)

    def test_scan_mode(self, separable_2d):
        X, y = separable_2d
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                          use_scan=True)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (16,)

    def test_save_load_roundtrip(self, separable_2d, tmp_path):
        X, y = separable_2d
        model = DKGLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        path = str(tmp_path / "dkglvq_ng.npz")
        model.save(path)
        loaded = DKGLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )
        assert loaded.gamma_ == pytest.approx(model.gamma_, rel=1e-4)
        np.testing.assert_allclose(
            np.asarray(model.sigmas_), np.asarray(loaded.sigmas_), rtol=1e-4
        )


# ============================================================================
# DKGRLVQ_NG Tests
# ============================================================================

class TestDKGRLVQ_NG:

    def test_fit_predict(self, separable_4d):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                           use_scan=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)
        acc = float(jnp.mean(preds == y))
        assert acc >= 0.70

    def test_loss_decreases(self, separable_4d):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=100, lr=0.01,
                           use_scan=False)
        model.fit(X, y)
        history = model.loss_history_
        assert float(history[-1]) < float(history[0])

    def test_iris_accuracy(self, iris_data):
        X, y = iris_data
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        acc = float(jnp.mean(preds == y))
        assert acc >= 0.80

    def test_gamma_decays(self, separable_4d):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        assert model.gamma_ is not None
        assert model.gamma_ < model._gamma_init_actual

    def test_gamma_bounded(self, separable_4d):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                           gamma_init=5.0, gamma_final=0.5)
        model.fit(X, y)
        assert model.gamma_ >= 0.5
        assert model.gamma_ < 5.0

    def test_sigmas_learned(self, separable_4d):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                           use_scan=False)
        model.fit(X, y)
        assert model.sigmas_ is not None
        assert model.sigmas_.shape == (4,)
        assert jnp.all(model.sigmas_ > 0)

    def test_sigma_init_fixed(self, separable_4d):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=30, lr=0.01,
                           sigma_init=1.0)
        model.fit(X, y)
        assert model.sigmas_ is not None

    def test_relevances_learned(self, separable_4d):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                           use_scan=False)
        model.fit(X, y)
        assert model.relevances_ is not None
        assert model.relevances_.shape == (4,)

    def test_relevance_profile_property(self, separable_4d):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                           use_scan=False)
        model.fit(X, y)
        profile = model.relevance_profile
        assert profile.shape == (4,)
        assert abs(float(jnp.sum(profile)) - 1.0) < 1e-5
        assert jnp.all(profile > 0)

    def test_kernel_bandwidths_property(self, separable_4d):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=20, lr=0.01)
        model.fit(X, y)
        bw = model.kernel_bandwidths
        assert bw.shape == (4,)
        assert jnp.all(bw >= model.sigma_min)

    def test_scan_mode(self, separable_4d):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                           use_scan=True)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30,)

    def test_save_load_roundtrip(self, separable_4d, tmp_path):
        X, y = separable_4d
        model = DKGRLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        path = str(tmp_path / "dkgrlvq_ng.npz")
        model.save(path)
        loaded = DKGRLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )
        assert loaded.gamma_ == pytest.approx(model.gamma_, rel=1e-4)
        np.testing.assert_allclose(
            np.asarray(model.relevances_), np.asarray(loaded.relevances_),
            rtol=1e-4
        )


# ============================================================================
# DKGMLVQ_NG Tests
# ============================================================================

class TestDKGMLVQ_NG:

    def test_fit_predict(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                           use_scan=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (16,)
        acc = float(jnp.mean(preds == y))
        assert acc >= 0.75

    def test_loss_decreases(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=100, lr=0.01,
                           use_scan=False)
        model.fit(X, y)
        history = model.loss_history_
        assert float(history[-1]) < float(history[0])

    def test_iris_accuracy(self, iris_data):
        X, y = iris_data
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=100, lr=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        acc = float(jnp.mean(preds == y))
        assert acc >= 0.80

    def test_gamma_decays(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        assert model.gamma_ is not None
        assert model.gamma_ < model._gamma_init_actual

    def test_gamma_bounded(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                           gamma_init=5.0, gamma_final=0.5)
        model.fit(X, y)
        assert model.gamma_ >= 0.5
        assert model.gamma_ < 5.0

    def test_omega_hat_learned(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                           use_scan=False)
        model.fit(X, y)
        assert model.omega_hat_ is not None
        assert model.omega_hat_.shape == (2, 2)

    def test_omega_hat_matrix_property(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=20, lr=0.01)
        model.fit(X, y)
        omega = model.omega_hat_matrix
        assert omega.shape == (2, 2)

    def test_lambda_hat_matrix_psd(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=20, lr=0.01)
        model.fit(X, y)
        lam = model.lambda_hat_matrix
        assert lam.shape == (2, 2)
        eigenvalues = jnp.linalg.eigvalsh(lam)
        assert jnp.all(eigenvalues >= -1e-6)

    def test_latent_dim(self, iris_data):
        X, y = iris_data
        model = DKGMLVQ_NG(n_prototypes_per_class=1, max_iter=20, lr=0.01,
                           latent_dim=2)
        model.fit(X, y)
        assert model.omega_hat_.shape == (4, 2)
        assert model.lambda_hat_matrix.shape == (4, 4)

    def test_numerical_stability(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=20, lr=0.01,
                           omega_hat_scale=0.01)
        model.fit(X, y)
        assert jnp.all(jnp.isfinite(model.prototypes_))
        assert jnp.all(jnp.isfinite(model.omega_hat_))

    def test_scan_mode(self, separable_2d):
        X, y = separable_2d
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01,
                           use_scan=True)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (16,)

    def test_save_load_roundtrip(self, separable_2d, tmp_path):
        X, y = separable_2d
        model = DKGMLVQ_NG(n_prototypes_per_class=2, max_iter=50, lr=0.01)
        model.fit(X, y)
        path = str(tmp_path / "dkgmlvq_ng.npz")
        model.save(path)
        loaded = DKGMLVQ_NG.load(path)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X)
        )
        assert loaded.gamma_ == pytest.approx(model.gamma_, rel=1e-4)
        np.testing.assert_allclose(
            np.asarray(model.omega_hat_), np.asarray(loaded.omega_hat_),
            rtol=1e-4
        )
